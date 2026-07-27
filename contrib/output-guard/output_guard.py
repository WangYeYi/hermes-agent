#!/usr/bin/env python3.10
"""
Output Guard —— 结构化覆盖校验器
===================================
在 Agent 回答后、返回用户前，检查是否覆盖了用户消息中的所有独立条目。

策略（3层降级）：
  L1: 规则引擎 — 专有名词精确匹配（0 token, ~1ms）
  L2: ONNX Embedding — bge-small-zh-v1.5 语义相似度（0 token, ~3ms, ~90MB）
  L3: LLM — 灰区语义判定（~300 tokens, ~1s）

集成方式：
  1. system_prompt_append: 注入结构化条目标记
  2. conversation_loop.py patch: 在 agent_response 后插入 guard 检查
  3. 或独立命令行: echo "reply" | python3 output_guard.py --items "q1" "q2"

依赖: onnxruntime, transformers, numpy
"""

import re
import sys
import json
import time
import os
import subprocess
import threading
from typing import Optional

import numpy as np

# ── 日志（JSONL，30 天保留）───────────────────────────────

_LOG_PATH = os.path.expanduser("~/.hermes/logs/output-guard.jsonl")
_LOG_MAX_AGE = 30 * 86400  # 30 天


def _log_guard_entry(entry: dict) -> None:
    """追加一条覆盖检查日志。自动清理 30 天前的旧记录。"""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        now = time.time()
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # 每次写入时顺便清理 30 天前的旧行（低频操作，开销可忽略）
        cutoff = now - _LOG_MAX_AGE
        if os.path.getmtime(_LOG_PATH) < now - 3600:
            _rotate_log(cutoff)
    except Exception:
        pass  # 日志失败不阻塞主流程


def _rotate_log(cutoff: float) -> None:
    """清理 cutoff 之前的日志行"""
    try:
        import tempfile
        tmp = _LOG_PATH + ".tmp"
        kept = 0
        with open(_LOG_PATH) as fin, open(tmp, "w") as fout:
            for line in fin:
                try:
                    entry = json.loads(line)
                    ts = entry.get("ts", "")
                    if ts and ts >= time.strftime("%Y-%m-%dT%H:%M:%S",
                                                  time.gmtime(cutoff)):
                        fout.write(line)
                        kept += 1
                except (json.JSONDecodeError, KeyError):
                    fout.write(line)  # 格式异常的行保留
                    kept += 1
        os.replace(tmp, _LOG_PATH)
    except Exception:
        pass


# ── L1: 规则引擎 ──────────────────────────────────────────

def rule_sure_pass(items: list[str], reply: str) -> set[int]:
    """
    确定覆盖：专有名词（英文、数字、特殊标识符）精确出现在回答中。
    100% 确定，不需要 embedding 或 LLM。
    """
    sure = set()
    for i, item in enumerate(items, 1):
        eng = re.findall(r'[a-zA-Z][a-zA-Z0-9._/-]{2,}', item)
        if eng and all(e.lower() in reply.lower() for e in eng):
            sure.add(i)
    return sure


def rule_sure_miss(items: list[str], reply: str) -> set[int]:
    """
    确定遗漏：专有名词全部不出现 + 短条目（≤4字）的每个中文字都不在回答中。
    高置信，不需要 embedding。
    """
    sure = set()
    for i, item in enumerate(items, 1):
        eng = re.findall(r'[a-zA-Z][a-zA-Z0-9._/-]{2,}', item)
        # 有专有名词但全部缺失 → 确定遗漏
        if eng and not any(e.lower() in reply.lower() for e in eng):
            sure.add(i)
            continue
        # 短条目（≤4个中文字）完全没有字符匹配 → 确定遗漏
        if not eng:
            chars = re.findall(r'[\u4e00-\u9fff]', item)
            if len(chars) <= 4 and not any(c in reply for c in chars):
                sure.add(i)
    return sure


# ── L2: Embedding (双模型 ONNX 投票) ─────────────────────
# bge-small-zh-v1.5 (512维, CLS) + bge-base-zh-v1.5 (768维, CLS)
# 双模型一致 → PASS/MISS；不一致 → 灰区送 L3

_SMALL_SESSION = None
_SMALL_TOKENIZER = None
_BASE_SESSION = None
_BASE_TOKENIZER = None
_LOCK = threading.Lock()


def _get_onnx_models():
    global _SMALL_SESSION, _SMALL_TOKENIZER, _BASE_SESSION, _BASE_TOKENIZER
    if _SMALL_SESSION is None:
        with _LOCK:
            if _SMALL_SESSION is None:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                for path, attr_sess, attr_tok in [
                    ("/root/.cache/hermes/onnx/bge-small-zh-v1.5", "_SMALL_SESSION", "_SMALL_TOKENIZER"),
                    ("/root/.cache/hermes/onnx/bge-base-zh-v1.5", "_BASE_SESSION", "_BASE_TOKENIZER"),
                ]:
                    sess = ort.InferenceSession(f"{path}/model.onnx", providers=["CPUExecutionProvider"])
                    tok = AutoTokenizer.from_pretrained(path)
                    globals()[attr_sess] = sess
                    globals()[attr_tok] = tok
    return (_SMALL_SESSION, _SMALL_TOKENIZER), (_BASE_SESSION, _BASE_TOKENIZER)


def _bge_encode(session, tokenizer, texts: list[str]) -> np.ndarray:
    """CLS pooling + L2 normalize → (N, dim)"""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    out = session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                              "token_type_ids": enc["token_type_ids"]})
    cls_emb = out[0][:, 0, :]
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    return cls_emb / norms


def embedding_check(items: list[str], reply: str, indices: list[int],
                    threshold: float = 0.45) -> dict[int, bool]:
    """双模型投票：返回 {index: True(覆盖)/False(遗漏)}"""
    if not indices:
        return {}
    (small_sess, small_tok), (base_sess, base_tok) = _get_onnx_models()
    items_subset = [items[i - 1] for i in indices]
    try:
        small_vecs = _bge_encode(small_sess, small_tok, items_subset)
        small_reply = _bge_encode(small_sess, small_tok, [reply])
        base_vecs = _bge_encode(base_sess, base_tok, items_subset)
        base_reply = _bge_encode(base_sess, base_tok, [reply])
    except Exception:
        return {idx: False for idx in indices}

    results = {}
    for j, idx in enumerate(indices):
        sim_small = float(np.dot(small_vecs[j], small_reply[0]))
        sim_base = float(np.dot(base_vecs[j], base_reply[0]))
        # 两票一致 → 采纳；不一致 → 取平均保守判定
        vote_small = sim_small >= threshold
        vote_base = sim_base >= threshold
        if vote_small == vote_base:
            results[idx] = vote_small
        else:
            results[idx] = (sim_small + sim_base) / 2 >= threshold
    return results


# ── L3: LLM fallback ────────────────────────────────────

def llm_fallback(items: list[str], reply: str, uncertain: list[int]) -> list[int]:
    """
    对 embedding 不确定的条目，调用 DeepSeek API 做最终判定。
    一次调用判定所有 uncertain 条目。
    """
    if not uncertain:
        return []
    
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        try:
            import yaml
            with open(os.path.expanduser("~/.hermes/config.yaml")) as f:
                cfg = yaml.safe_load(f)
            api_key = cfg.get("providers", {}).get("deepseek", {}).get("api_key", "")
        except:
            pass
    
    if not api_key:
        # 无 API key → 所有 uncertain 判为遗漏（宁可误报）
        return uncertain
    
    items_text = "\n".join(f"  [{i}] {items[i-1]}" for i in uncertain)
    prompt = (
        f"你是输出覆盖校验器。判断以下{len(uncertain)}个条目是否被AI回答覆盖。\n"
        f"只输出JSON数组，格式：[{{\"item\": 序号, \"covered\": true/false, \"reason\": \"理由\"}}]\n\n"
        f"条目列表：\n{items_text}\n\nAI回答：\n{reply}"
    )
    
    try:
        payload = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
        })
        result = subprocess.run(
            ["curl", "-s", "--max-time", "10",
             "https://api.deepseek.com/v1/chat/completions",
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=15
        )
        resp = json.loads(result.stdout)
        content = resp["choices"][0]["message"]["content"]
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            judgments = json.loads(match.group())
            return [j["item"] for j in judgments if not j["covered"]]
    except:
        pass
    
    return uncertain  # 失败时保守：全部判为遗漏


# ── 主入口 ───────────────────────────────────────────────

def check_coverage(items: list[str], reply: str,
                   embedding_threshold: float = 0.45) -> dict:
    """
    完整的覆盖检查流程。
    
    返回:
        {
            "verdict": "PASS" | "MISSING",
            "missed": [idx, ...],
            "cost": {"rule": 0, "embedding": 0, "llm": N_tokens},
            "timing_ms": {"rule": N, "embedding": N, "llm": N}
        }
    """
    t0 = time.time()
    cost = {"rule": 0, "embedding": 0, "llm": 0}
    timing = {"rule": 0, "embedding": 0, "llm": 0}
    
    n = len(items)
    all_indices = set(range(1, n + 1))
    
    # L1: 规则引擎
    t1 = time.time()
    sure_pass = rule_sure_pass(items, reply)
    sure_miss = rule_sure_miss(items, reply)
    timing["rule"] = (time.time() - t1) * 1000
    
    remaining = all_indices - sure_pass - sure_miss
    
    if not remaining:
        result = {
            "verdict": "PASS" if not sure_miss else "MISSING",
            "missed": sorted(sure_miss),
            "cost": cost, "timing": timing,
            "detail": {"sure_pass": sorted(sure_pass), "sure_miss": sorted(sure_miss)}
        }
        _log_guard_entry(_build_log(items, reply, embedding_threshold,
                                    sure_pass, sure_miss, {}, False, result))
        return result
    
    # L2: Embedding（只对 remaining 做）
    t2 = time.time()
    emb_results = embedding_check(items, reply, list(remaining), embedding_threshold)
    timing["embedding"] = (time.time() - t2) * 1000
    
    emb_pass = {idx for idx, covered in emb_results.items() if covered}
    emb_miss = {idx for idx, covered in emb_results.items() if not covered}
    
    # 检查是否有灰区（接近阈值的）
    truly_uncertain = []
    final_missed = list(sure_miss | emb_miss)
    
    # L3: LLM fallback（当 embedding 判定全部遗漏但实际可能覆盖时）
    if emb_miss and len(emb_miss) >= len(remaining) * 0.7:
        # embedding 判定大部分遗漏 → 可能阈值太严，LLM 复核
        t3 = time.time()
        llm_uncertain = list(emb_miss)
        llm_results = llm_fallback(items, reply, llm_uncertain)
        timing["llm"] = (time.time() - t3) * 1000
        cost["llm"] = 300
        
        # LLM 判覆盖的 → 从遗漏中移除
        llm_pass = set(llm_uncertain) - set(llm_results)
        final_missed = sorted(set(final_missed) - llm_pass)
    
    result = {
        "verdict": "PASS" if not final_missed else "MISSING",
        "missed": sorted(final_missed),
        "cost": cost, "timing": timing,
        "detail": {
            "sure_pass": sorted(sure_pass),
            "sure_miss": sorted(sure_miss),
            "emb_pass": sorted(emb_pass),
            "emb_miss": sorted(emb_miss),
        }
    }
    _log_guard_entry(_build_log(items, reply, embedding_threshold,
                                sure_pass, sure_miss, emb_results,
                                bool(cost.get("llm", 0)), result))
    return result


def _build_log(items, reply, threshold, sure_pass, sure_miss,
               emb_results, l3_used, result):
    """构建标准化的 JSONL 日志条目"""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    entry = {
        "ts": ts,
        "session": os.environ.get("HERMES_SESSION_ID", ""),
        "model": "stella-base-zh-v3-1792d",
        "threshold": threshold,
        "items": [{"idx": i, "text": t[:200]} for i, t in enumerate(items, 1)],
        "reply_preview": reply[:500],
        "l1": {"sure_pass": sorted(sure_pass), "sure_miss": sorted(sure_miss)},
        "l2": {
            "source": "onnx",
            "scores": [{"idx": idx, "covered": covered}
                       for idx, covered in sorted(emb_results.items())],
        },
        "l3_fallback": l3_used,
        "missed": result["missed"],
        "verdict": result["verdict"],
        "total_timing_ms": round(sum(result["timing"].values()), 1) if "timing" in result else 0,
    }
    return entry


def split_user_message(text: str) -> list[str]:
    """A层：提问词匹配 + 句法拆分 → 拆分独立条目"""
    numbered = re.findall(
        r'(?:^|\n)\s*(?:[(（]?\d+[)）]|问题\d+|Q\d+|第\d+[条个点])[\s.、:：]+\s*(.+?)(?=\n\s*(?:[(（]?\d+[)）]|问题\d+|Q\d+|第\d+[条个点])|\Z)',
        text, re.DOTALL)
    if len(numbered) >= 2: return [t.strip() for t in numbered]
    segments = re.split(r'(?<=[。！？；])\s*|\n+', text)
    segments = [s.strip().rstrip('。！？；，,') for s in segments if len(s.strip().rstrip('。！？；，,')) >= 2]
    if len(segments) < 2:
        sub = re.split(r'(?:还有|另外|以及|另外问|顺便|同时|此外|再加上|最后|然后|而且)', text)
        sub = [s.strip().rstrip('，,') for s in sub if len(s.strip()) > 2]
        if len(sub) >= 2: return sub
        implicit = _split_implicit_multi(text)
        if len(implicit) >= 2: return implicit
        qsplit = _split_question_markers(text)
        return qsplit if len(qsplit) >= 2 else [text.strip()]
    result = []
    for seg in segments:
        sub = re.split(r'(?:还有|另外|以及|另外问|顺便问|同时|此外|再加上)', seg)
        for s in sub:
            s = s.strip()
            if len(s) >= 2: result.append(s)
    if len(result) >= 2: return result
    implicit = _split_implicit_multi(text)
    if len(implicit) >= 2: return implicit
    qsplit = _split_question_markers(text)
    return qsplit if len(qsplit) >= 2 else [text.strip()]


def _split_question_markers(text: str) -> list[str]:
    """拆分逗号分隔的隐性提问：'A，是不是B，能不能C' → ['A', '是不是B', '能不能C']"""
    markers = ['是不是','能不能','有没有','为什么','怎么','你确定','我感觉','检查一下','看一下','分析一下','还能']
    positions = []
    for m in markers:
        pos = 0
        while True:
            idx = text.find(m, pos)
            if idx == -1: break
            positions.append(idx)
            pos = idx + len(m)
    if not positions: return [text.strip()]
    positions = sorted(set(positions))
    result = []; prev = 0
    for pos in positions:
        if pos > prev:
            chunk = text[prev:pos].strip().rstrip('，,')
            if len(chunk) >= 2: result.append(chunk)
        prev = pos
    if prev < len(text):
        chunk = text[prev:].strip().rstrip('，,')
        if len(chunk) >= 2: result.append(chunk)
    if len(result) >= 2:
        merged = []; i = 0
        while i < len(result):
            if i+1 < len(result) and (len(result[i]) < 6 or result[i].rstrip()[-1] in '你我他'):
                merged.append(result[i] + result[i+1]); i += 2
            else: merged.append(result[i]); i += 1
        result = merged
    return result if len(result) >= 2 else [text.strip()]


def _split_implicit_multi(text: str) -> list[str]:
    """拆分隐式多问句：'A和B怎么样' → ['A怎么样', 'B怎么样']"""
    lo, hi = '\u4e00', '\u9fff'
    particles = set('的了着过之地得')
    for i, ch in enumerate(text):
        if ch not in '和与及':
            continue
        j, cnt = i - 1, 0
        while j >= 0 and lo <= text[j] <= hi and text[j] not in particles and cnt < 4:
            j -= 1
            cnt += 1
        a = text[j + 1:i]
        k, cnt2 = i + 1, 0
        while k < len(text) and lo <= text[k] <= hi and text[k] not in particles and cnt2 < 4:
            k += 1
            cnt2 += 1
        b = text[i + 1:k]
        if len(a) < 1 or len(b) < 1:
            continue
        prefix = text[:j + 1]
        suffix = text[k:]
        if (a != b and a not in '你我他她它' and b not in '你我他她它'
                and len(prefix + a + suffix) >= 2
                and len(prefix + b + suffix) >= 2):
            return [f"{prefix}{a}{suffix}".strip(),
                    f"{prefix}{b}{suffix}".strip()]
    return [text.strip()]


def generate_structured_prompt(items: list[str]) -> str:
    """生成注入 system prompt 的结构化条目标记"""
    if len(items) <= 1:
        return ""
    lines = [f"[STRUCTURED_ITEMS: {len(items)} 条 — 必须逐一回复，不得遗漏]"]
    for i, item in enumerate(items, 1):
        lines.append(f"  ITEM_{i}: {item}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 output_guard.py check '<reply>' --items 'q1' 'q2' ...")
        print("       python3 output_guard.py split '<user_message>'")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "split":
        text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
        items = split_user_message(text)
        prompt = generate_structured_prompt(items)
        print(json.dumps({"items": items, "count": len(items), "prompt": prompt},
                        ensure_ascii=False, indent=2))
    
    elif cmd == "check":
        # 从 stdin 读取 reply，从 args 读取 items
        reply = sys.stdin.read().strip()
        items = []
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--items" and i + 1 < len(sys.argv):
                items.append(sys.argv[i + 1])
                i += 2
            elif not sys.argv[i].startswith("--"):
                items.append(sys.argv[i])
                i += 1
            else:
                i += 1
        
        result = check_coverage(items, reply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
