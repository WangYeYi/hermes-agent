#!/bin/bash
# Hermes IME 修复 — prompt_toolkit raw_mode 保留 IEXTEN
# 用法: bash /root/.hermes/scripts/hermes-ime-fix.sh
#
# 问题: prompt_toolkit 的 raw_mode 禁用了 IEXTEN，导致 ConPTY/WSL
#       在焦点切换/截图/IME状态变更后停止转发中文IME字节。
# 修复: _patch_lflag 中移除 IEXTEN，保留其开启状态。
#       IEXTEN=ON 安全，因为 ICANON=OFF 已阻止内核拦截特殊字符。

BASE="/root/.hermes/hermes-agent/venv/lib/python3.11/site-packages/prompt_toolkit/input"
FILE="$BASE/vt100.py"

# 检查是否已打过补丁
if grep -q "IEXTEN kept ON" "$FILE" 2>/dev/null; then
    echo "✓ IEXTEN 补丁已存在，跳过"
    exit 0
fi

# 备份原文件
cp "$FILE" "$FILE.bak-$(date +%Y%m%d)"
echo "✓ 已备份 $FILE"

# 打补丁
python3 -c "
import sys
with open('$FILE', 'r') as f:
    content = f.read()

old = 'return attrs & ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)'
new = '''        # NOTE: IEXTEN kept ON — ConPTY/WSL IME requires it.
        # Disabling IEXTEN causes ConPTY to stop forwarding IME bytes after
        # focus switch / screenshot / IME state change in another window.
        # IEXTEN=ON is safe because ICANON=OFF prevents LNEXT/WERASE/REPRINT
        # from being interpreted by the kernel line discipline.
        return attrs & ~(termios.ECHO | termios.ICANON | termios.ISIG)'''

if old in content:
    content = content.replace(old, new)
    with open('$FILE', 'w') as f:
        f.write(content)
    print('✓ IEXTEN 补丁已应用')
else:
    print('⚠ 未找到原始代码，可能已修改或版本不匹配')
"

# 验证语法
python3 -m py_compile "$FILE" && echo "✓ 语法验证通过"
