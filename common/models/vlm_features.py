"""h_tokens extraction — the ONLY function used for both caching (01_cache_h.py) and online rollout.

h_tokens (16, d_vlm) = 12 image tokens (8x8 image-token grid -> adaptive avg pool 3x4)
                     + 4 text tokens   (tokens after the closing image marker, i.e. instruction +
                                         end tokens; they attend to image AND instruction under causal
                                         attention -> adaptive avg pool 1D to 4)
Inputs are RAW images (hdf5 `agentview_rgb` or env `agentview_image`); `to_model_image` is applied here
so caching and rollout share preprocessing (CLAUDE.md pitfalls 1, 2, 4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.data.libero_dataset import to_model_image

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def preprocess_images(image_processor, raw_images: np.ndarray) -> dict:
    """CPU part (no model; safe in DataLoader workers): raw uint8 (B,H,W,3) -> pixel tensors."""
    from PIL import Image

    imgs = [[Image.fromarray(x)] for x in to_model_image(np.asarray(raw_images))]
    out = image_processor(images=imgs, return_tensors="pt")
    return {"pixel_values": out["pixel_values"], "pixel_attention_mask": out["pixel_attention_mask"]}


class VLMFeatureExtractor:
    def __init__(self, model_name: str = DEFAULT_MODEL, h_layer: int = -1, img_pool=(3, 4), n_txt_pool: int = 4,
                 dtype: str = "bfloat16", device: str = "cuda", do_image_splitting: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.h_layer = h_layer
        self.img_pool = tuple(img_pool)
        self.n_txt_pool = n_txt_pool
        self.device = device
        self.do_image_splitting = do_image_splitting
        self.proc = AutoProcessor.from_pretrained(model_name)
        # Original PIL (LANCZOS) image processor; image splitting off -> one 512x512 view = 64 image tokens.
        self.proc.image_processor.do_image_splitting = do_image_splitting
        self._ids_cache: dict = {}
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=getattr(torch, dtype)).to(device).eval()
        self.image_token_id = self.model.config.image_token_id
        self.fake_token_id = self.proc.tokenizer.convert_tokens_to_ids("<fake_token_around_image>")
        self.d_vlm = self.model.config.text_config.hidden_size
        self.n_tokens = self.img_pool[0] * self.img_pool[1] + n_txt_pool

    def config(self) -> dict:
        return {"model_name": self.model_name, "h_layer": self.h_layer, "img_pool": list(self.img_pool),
                "n_txt_pool": self.n_txt_pool, "do_image_splitting": self.do_image_splitting,
                "d_vlm": self.d_vlm, "n_tokens": self.n_tokens, "image_transform": "to_model_image"}

    def _prompt(self, instruction: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=False)

    def preprocess(self, raw_images: np.ndarray) -> dict:
        return preprocess_images(self.proc.image_processor, raw_images)

    def _text_inputs(self, instruction: str):
        if instruction not in self._ids_cache:
            from PIL import Image

            dummy = Image.fromarray(np.zeros((128, 128, 3), np.uint8))
            t = self.proc(text=[self._prompt(instruction)], images=[[dummy]], return_tensors="pt")
            self._ids_cache[instruction] = (t["input_ids"], t["attention_mask"])
        return self._ids_cache[instruction]

    @torch.no_grad()
    def encode(self, raw_images: np.ndarray, instruction: str, return_debug: bool = False):
        """raw_images: (B, H, W, 3) uint8, all with the same instruction. Returns (B, 16, d_vlm) float16 tensor."""
        return self.encode_preprocessed(self.preprocess(raw_images), instruction, return_debug)

    @torch.no_grad()
    def encode_preprocessed(self, pix: dict, instruction: str, return_debug: bool = False):
        B = pix["pixel_values"].shape[0]
        input_ids, attn = self._text_inputs(instruction)
        inputs = {
            "input_ids": input_ids.expand(B, -1).to(self.device),
            "attention_mask": attn.expand(B, -1).to(self.device),
            "pixel_values": pix["pixel_values"].to(self.device, self.model.dtype, non_blocking=True),
            "pixel_attention_mask": pix["pixel_attention_mask"].to(self.device, non_blocking=True),
        }
        ids = inputs["input_ids"][0]
        img_idx = (ids == self.image_token_id).nonzero().squeeze(-1)
        fake_idx = (ids == self.fake_token_id).nonzero().squeeze(-1)
        gh, gw = self.img_pool
        n_img = img_idx.numel()
        side = int(round(n_img ** 0.5))
        assert side * side == n_img, f"image tokens {n_img} not square (image splitting on?)"
        txt_start = int(fake_idx.max().item()) + 1
        assert txt_start > int(img_idx.max().item())
        out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[self.h_layer].float()  # (B, L, D)
        B, L, D = hs.shape
        img = hs[:, img_idx, :].reshape(B, side, side, D).permute(0, 3, 1, 2)  # (B, D, s, s)
        img = F.adaptive_avg_pool2d(img, (gh, gw)).flatten(2).transpose(1, 2)  # (B, gh*gw, D)
        txt = hs[:, txt_start:, :].transpose(1, 2)  # (B, D, Lt)
        txt = F.adaptive_avg_pool1d(txt, self.n_txt_pool).transpose(1, 2)  # (B, n_txt, D)
        h = torch.cat([img, txt], dim=1).half()
        if return_debug:
            tok = self.proc.tokenizer
            return h, {"seq_len": L, "image_token_range": [int(img_idx.min()), int(img_idx.max())],
                       "n_image_tokens": n_img, "text_range": [txt_start, L - 1],
                       "text_tokens": tok.decode(ids[txt_start:])}
        return h
