"""Model IDs (CLAUDE.md §2) -> nn.Module."""
from __future__ import annotations

from common.models.ae_transformer import ActionExpert

CONDS = {"M1_h": ["h"], "M2_O": ["O"], "M3_hO": ["h", "O"]}


def build_model(model_id: str, m: dict):
    kw = dict(H=m["H"], action_dim=m["action_dim"], d=m["d"], n_heads=m["n_heads"], d_vlm=m["d_vlm"],
              n_h=m["n_h"], proprio_dim=m["proprio_dim"], pretrained_wrist=m.get("pretrained_wrist", True))
    if model_id in CONDS:
        return ActionExpert(CONDS[model_id], n_layers=m["n_layers"], dropout=m.get("dropout", 0.0), **kw)
    if model_id == "M4_moe":
        from experiment1.models.moe import MoEAE

        return MoEAE(n_layers=m["moe_n_layers"], router_hidden=m.get("router_hidden", 256), **kw)
    raise ValueError(model_id)


def uses(model_id: str) -> set:
    return {"M1_h": {"h"}, "M2_O": {"O"}, "M3_hO": {"h", "O"}, "M4_moe": {"h", "O"}}[model_id]
