#!/bin/bash
# 恢复 prompt_toolkit 到原始干净版本
# 用法: bash /root/.hermes/scripts/hermes-ime-restore.sh
# 注：2026-07-16 更新后 prompt_toolkit 已通过 pip 重装为干净版本。
# 此脚本作为备用，在 venv 重建后需重新打入 IME 补丁时使用。

HERMES_ROOT="/root/.hermes/hermes-agent"
echo "→ 通过 pip 恢复 prompt_toolkit 到干净版本..."
$HERMES_ROOT/venv/bin/pip install --force-reinstall --no-deps prompt-toolkit 2>&1 | tail -3
echo ""
echo "验证："
python3 -c "
import prompt_toolkit
print(f'  prompt_toolkit version: {prompt_toolkit.__version__}')
from prompt_toolkit.input.vt100 import raw_mode
import inspect
src = inspect.getsource(raw_mode._patch_lflag)
print(f'  _patch_lflag has IEXTEN: {\"IEXTEN\" in src}')
"
