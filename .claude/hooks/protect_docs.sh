#!/usr/bin/env bash
# CLAUDE.md, docs/plan/** 을 셸 명령으로 수정하려는 Bash 호출을 차단한다.
cmd=$(jq -r '.tool_input.command // ""')
if printf '%s' "$cmd" | grep -qE 'CLAUDE\.md|docs/plan'; then
  if printf '%s' "$cmd" | grep -qE '(^|[^0-9&])>{1,2}|\b(sed[^|;&]*-i|perl[^|;&]*-i|tee|mv|cp|rm|rmdir|truncate|chmod|chown|ln|touch|dd|install|rsync|git (checkout|restore|rm|mv))\b|python|open\(|write'; then
    jq -n '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:"CLAUDE.md와 docs/plan/**은 보호 파일입니다. Claude는 수정할 수 없습니다 (사용자가 직접 수정)."}}'
    exit 0
  fi
fi
exit 0
