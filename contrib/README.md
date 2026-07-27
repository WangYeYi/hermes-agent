# 本地功能合集

这些文件不受 `hermes update` 影响（在源码树外），可独立维护。

## 部署路径

| 源路径 | → 部署路径 |
|--------|-----------|
| `contrib/flash/` | `~/.hermes/scripts/` + `E:\0AIHermes\scripts\flash\` |
| `contrib/hooks/` | `~/.hermes/scripts/` |
| `contrib/output-guard/` | `~/.hermes/scripts/` |
| `contrib/embedding/` | `~/.hermes/scripts/` |
| `contrib/ime/` | `~/.hermes/scripts/` |
| `contrib/token-stats/` | `~/.hermes/scripts/` |

## config.yaml 需要的 hook 注册

```yaml
hooks:
  pre_approval_request:
    - command: python3 /root/.hermes/scripts/hook-approval-flash-start.py
      timeout: 15
  post_approval_response:
    - command: python3 /root/.hermes/scripts/hook-approval-flash-stop.py
      timeout: 15
```
