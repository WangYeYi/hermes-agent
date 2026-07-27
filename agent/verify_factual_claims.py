#!/usr/bin/env python3
"""
事实验证 dispatch 层钩子 — 回答生成后、发给用户前运行。

用 MiniLM 做轻量二分分类（claim / not claim），命中 claim 后正则提取
实体编号 → 调外部命令验证 → 不匹配则返回纠正 nudge。

与 output-guard 正交：output-guard 检查覆盖遗漏，本脚本检查编造。
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

MINILM_URL = "http://localhost:5199/embed"
THRESHOLD = 0.55  # 相似度 >= 此值判为 claim

# ── 参考句（claim 模板）──
CLAIM_SENTENCES = [
    "PR #12345 has been closed",
    "Issue #67890 was fixed",
    "这个 PR 已经 merge 了",
    "上游已经修了这个 bug",
    "版本号是 v1.2.3",
    "价格是 ¥6 每百万 token",
    "配置是 timeout=60",
    "那个文件不存在",
    "那个 commit 已经合入了",
    "该功能已被移除",
]

# ── 参考句（not-claim 模板）──
NON_CLAIM_SENTENCES = [
    "这个方案可以",
    "让我检查一下",
    "好的，我来做",
    "这是一个好问题",
    "收到，执行中",
    "完成，总结如下",
    "我会处理的",
    "确认，继续",
    "对比结果如下",
    "测试通过了",
]


def _miniLM_embed(texts: list[str]) -> list[list[float]] | None:
    """调 MiniLM HTTP 服务获取 embedding。"""
    try:
        data = json.dumps({"texts": texts}).encode("utf-8")
        req = urllib.request.Request(
            MINILM_URL, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return result.get("embeddings")
    except Exception:
        return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _classify(text: str, claim_embs: list[list[float]], nonclaim_embs: list[list[float]]) -> tuple[bool, float]:
    """返回 (is_claim, max_similarity)。"""
    my_emb = _miniLM_embed([text])
    if not my_emb or not my_emb[0]:
        return False, 0.0
    my = my_emb[0]

    claim_sim = max(_cosine_sim(my, ce) for ce in claim_embs) if claim_embs else 0.0
    nonclaim_sim = max(_cosine_sim(my, ne) for ne in nonclaim_embs) if nonclaim_embs else 0.0
    return claim_sim > nonclaim_sim and claim_sim >= THRESHOLD, claim_sim


# ── 实体提取 + 验证 ────────────────────────────────────────────

def _verify_github_ref(sentence: str) -> list[dict]:
    """提取 PR/issue 引用并验证状态。返回 [{"claim": ..., "actual": ..., "ok": bool}]。"""
    corrections = []
    # PR #\d+ 或 pull/\d+
    for m in re.finditer(r'(?:PR|pull)\s*#?(\d+)', sentence):
        num = m.group(1)
        try:
            r = subprocess.run(
                ["gh", "pr", "view", num, "--repo", "NousResearch/hermes-agent",
                 "--json", "state,title", "--jq", "{state: .state, title: .title}"],
                capture_output=True, timeout=10, text=True
            )
            if r.returncode == 0:
                actual = json.loads(r.stdout)
            else:
                # 可能是 issue
                r2 = subprocess.run(
                    ["gh", "issue", "view", num, "--repo", "NousResearch/hermes-agent",
                     "--json", "state,title", "--jq", "{state: .state, title: .title}"],
                    capture_output=True, timeout=10, text=True
                )
                actual = json.loads(r2.stdout) if r2.returncode == 0 else None

            if actual:
                # 检查回答中声称的状态是否匹配
                claimed_closed = bool(re.search(r'CLOSED|已关|已合|已修|已 merge|merged|closed',
                                                 sentence, re.IGNORECASE))
                actual_closed = actual["state"] in ("CLOSED", "MERGED")
                if claimed_closed != actual_closed:
                    corrections.append({
                        "entity": f"#{num}",
                        "claim": sentence.strip()[:120],
                        "actual": f"state={actual['state']}, title={actual['title'][:80]}",
                        "ok": False,
                    })
        except Exception:
            pass
    return corrections


def verify_response(response_text: str) -> list[dict]:
    """主入口：扫描回答，返回需要纠正的 claim 列表。"""
    # 1. 拆句
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', response_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return []

    # 2. 获取参考 embedding（缓存）
    all_refs = CLAIM_SENTENCES + NON_CLAIM_SENTENCES
    all_embs = _miniLM_embed(all_refs)
    if not all_embs:
        return []
    claim_embs = all_embs[:len(CLAIM_SENTENCES)]
    nonclaim_embs = all_embs[len(CLAIM_SENTENCES):]

    # 3. 获取句子 embedding（批量）
    sent_embs = _miniLM_embed(sentences)
    if not sent_embs:
        return []

    # 4. 分类 + 验证
    corrections = []
    for i, (sent, emb) in enumerate(zip(sentences, sent_embs)):
        if not emb:
            continue
        claim_sim = max(_cosine_sim(emb, ce) for ce in claim_embs) if claim_embs else 0.0
        nonclaim_sim = max(_cosine_sim(emb, ne) for ne in nonclaim_embs) if nonclaim_embs else 0.0
        is_claim = claim_sim > nonclaim_sim and claim_sim >= THRESHOLD

        if is_claim:
            # GitHub 验证
            gh_corrections = _verify_github_ref(sent)

            # 只有验证出真正不匹配的才报；提取不到实体且无纠错的标记"需人工"
            if gh_corrections:
                corrections.extend(gh_corrections)
            elif not re.search(r'(?:#|PR|pull|issue)\s*\d+', sent, re.IGNORECASE):
                # 没有编号的模糊断言（如"上游已经修了"）→ 需人工
                corrections.append({
                    "entity": "?",
                    "claim": sent[:120],
                    "actual": "无法自动验证，请确认",
                    "ok": False,
                })
            # else: 有编号且验证通过 → 不报

    return corrections


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = sys.stdin.read()
    corr = verify_response(text)
    print(json.dumps(corr, ensure_ascii=False, indent=2))
