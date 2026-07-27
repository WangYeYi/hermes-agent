#!/usr/bin/env python3
"""
分用户 / 分模型 / 分层级 Token 消耗统计 v3
- 人民币计价（美元按汇率转换）
- 中文计量单位（千、万、亿）
- 终端表格输出 / JSON 导出

用法:
  python3 token_stats.py [天数]              表格输出
  python3 token_stats.py detail [天数]        分模型明细
  python3 token_stats.py json [天数]          JSON
  python3 token_stats.py models               模型定价
  python3 token_stats.py collect              采集全部数据到 stats.db
  python3 token_stats.py catchup              启动时自动补齐
  python3 token_stats.py count [文本]          本地分词器计数
  python3 token_stats.py verify [天数]         对比本地分词器 vs API 报告
  python3 token_stats.py calibrate <file>     导入 DeepSeek 官方导出 CSV/ZIP 对账

"""
import sqlite3, json, sys, os
from datetime import datetime, timedelta, date

DB = "/root/.hermes/state.db"
STATS_DB = "/root/.hermes/data/token-stats.db"
TOKENIZER_PATH = "/root/.hermes/data/deepseek-tokenizer/tokenizer.json"
USD_TO_CNY = 7.25  # 仅用于参考，PRICING 已是人民币计价

# 分词器惰性加载
_tokenizer = None

def _get_tokenizer():
    """惰性加载 DeepSeek V3 tokenizer（Rust 原生 tokenizers 库）"""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    return _tokenizer

def count_tokens_local(text):
    """本地计算文本 token 数（不含 special tokens）"""
    t = _get_tokenizer()
    return len(t.encode(text).ids)

# Linux 用户名 → 显示名
LINUX_USER_MAP = {
    "user001": "afeng",
    "jinlong": "jinlong",
}

# 多用户 state.db → 终端用户标签（自动扫描 /home/*/ 和 /root）
def _scan_state_dbs():
    dbs = {}
    for base in ["/root", "/home"]:
        try:
            for name in os.listdir(base):
                path = os.path.join(base, name, ".hermes", "state.db")
                if os.path.exists(path):
                    label = LINUX_USER_MAP.get(name, name)
                    dbs[os.path.realpath(path)] = label
        except Exception:
            pass
    root_db = "/root/.hermes/state.db"
    if root_db not in dbs:
        dbs[root_db] = "root"
    return dbs

STATE_DBS = _scan_state_dbs()

USER_MAP = {
    "ou_6135b53598e992d5023d46deeadab0ad": "afeng",
    "ou_3d0471ae7a086fcf06915535a086dcea": "jinlong",
}

PRICING = {
    # 当前使用（¥/百万tokens）
    # 数据源: https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    ("deepseek", "deepseek-v4-flash"):   {"input": 1, "output": 2, "cache_read": 0.02},
    ("deepseek", "deepseek-v4-pro"):     {"input": 3, "output": 6, "cache_read": 0.025},
    # 新增模型在此添加: ("provider", "model"): {"input": x, "output": y, ...}
    # 所有价格单位为 ¥/百万tokens
}

ONE_MIL = 1_000_000
ONE_K = 1_000
ONE_YI = 100_000_000


def resolve_user(uid, terminal_user=None):
    """open_id → 飞书用户名；CLI 会话按 state.db 归属标记"""
    if uid in USER_MAP:
        return USER_MAP[uid]
    if uid is None or uid == "cli":
        return terminal_user or "terminal"
    return uid[:12]


def resolve_pricing(billing_provider, model):
    if not billing_provider or not model: return None
    bp = billing_provider.strip().lower()
    m = model.strip().lower()
    key = (bp, m)
    if key in PRICING: return PRICING[key]
    if "/" in m:
        key2 = (bp, m.split("/")[-1])
        if key2 in PRICING: return PRICING[key2]
    if bp == "deepseek":
        for (p, pm), price in PRICING.items():
            if p == "deepseek" and pm in m: return price
    return None


def compute_cost(input_tok, output_tok, cache_read, cache_write, pricing):
    cost = 0.0
    if pricing:
        cost += (input_tok or 0) * pricing.get("input", 0) / ONE_MIL
        cost += (output_tok or 0) * pricing.get("output", 0) / ONE_MIL
        cost += (cache_read or 0) * pricing.get("cache_read", 0) / ONE_MIL
        cost += (cache_write or 0) * pricing.get("cache_write", 0) / ONE_MIL
    return cost


def usd_to_cny(usd):
    return usd * USD_TO_CNY


def fmt_cn(n):
    """中文计量：低于千无单位，千、万、亿"""
    if n is None: return "0"
    n = int(n)
    if n >= ONE_YI:
        return f"{n/ONE_YI:.2f}亿"
    if n >= 10000:
        return f"{n/10000:.1f}万"
    if n >= 1000:
        return f"{n/1000:.1f}千"
    return str(n)





def fmt_money_cny(usd):
    """美元→人民币，>=1元显示两位小数，<1元显示四位"""
    cny = usd_to_cny(usd)
    if cny >= 1:
        return f"¥{cny:.2f}"
    if cny >= 0.01:
        return f"¥{cny:.4f}"
    return f"¥{cny:.6f}"


# ── 数据查询 ──────────────────────────────────────────────────
def query_detail(days, since_ts=None, until_ts=None):
    """按 (用户, 模型, provider) 分组查询，扫描所有用户 state.db"""
    if since_ts is None:
        since_ts = (datetime.now() - timedelta(days=days)).timestamp()

    all_rows = []
    for db_path, terminal_label in STATE_DBS.items():
        try:
            db = sqlite3.connect(db_path)
            if until_ts is not None:
                rows = db.execute("""
                    SELECT COALESCE(user_id, 'cli'), model, billing_provider,
                           COUNT(*), SUM(message_count),
                           SUM(input_tokens), SUM(output_tokens),
                           SUM(cache_read_tokens), SUM(cache_write_tokens),
                           SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0))
                    FROM sessions WHERE started_at >= ? AND started_at < ?
                    GROUP BY 1, 2, 3 ORDER BY 10 DESC
                """, (since_ts, until_ts)).fetchall()
            else:
                rows = db.execute("""
                    SELECT COALESCE(user_id, 'cli'), model, billing_provider,
                           COUNT(*), SUM(message_count),
                           SUM(input_tokens), SUM(output_tokens),
                           SUM(cache_read_tokens), SUM(cache_write_tokens),
                           SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0))
                    FROM sessions WHERE started_at >= ?
                    GROUP BY 1, 2, 3 ORDER BY 10 DESC
                """, (since_ts,)).fetchall()
            db.close()
            # 给每行附上终端来源标签
            for r in rows:
                all_rows.append((r, terminal_label))
        except Exception:
            pass  # state.db 不存在或无法读取则跳过

    # 合并同 (user, model) 不同 provider 的行（修复 provider=NULL 覆盖 provider=deepseek 的 bug）
    merged = {}
    for r, terminal_label in all_rows:
        user = resolve_user(r[0], terminal_label)
        model = r[1] or "unknown"
        bp = r[2] or ""
        key = (user, model)
        if key not in merged:
            merged[key] = {
                "user": user, "model": model,
                "providers": set(), "sessions": 0, "msgs": 0,
                "input_cache_miss": 0, "cache_read_hit": 0,
                "output": 0, "cache_write": 0,
                "cost_hermes_est": 0.0,
            }
        m = merged[key]
        m["providers"].add(bp)
        m["sessions"] += r[3]
        m["msgs"] += r[4] or 0
        m["input_cache_miss"] += r[5] or 0
        m["cache_read_hit"] += r[7] or 0
        m["output"] += r[6] or 0
        m["cache_write"] += r[8] or 0
        m["cost_hermes_est"] += r[9] or 0

    result = []
    for (user, model), m in merged.items():
        # 优先选有意义的 provider（非空），回退到第一个
        providers = m["providers"] - {""}
        bp = next(iter(providers)) if providers else (next(iter(m["providers"])) if m["providers"] else "")
        inp, out, cr, cw = m["input_cache_miss"], m["output"], m["cache_read_hit"], m["cache_write"]
        pricing = resolve_pricing(bp, model)
        calc_cost = compute_cost(inp, out, cr, cw, pricing)  # ¥
        total_input = inp + cr + (cw or 0)
        cache_hit_rate = (cr / total_input * 100) if total_input > 0 else 0

        result.append({
            "user": user,
            "model": model,
            "provider": bp,
            "sessions": m["sessions"],
            "msgs": m["msgs"],
            "input_cache_miss": inp,
            "cache_read_hit": cr,
            "output": out,
            "cache_write": cw,
            "total_input": total_input,
            "total_tokens": total_input + out,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "cost_cny": round(calc_cost, 6),
            "cost_hermes_est": round(m["cost_hermes_est"], 6),
            "pricing_used": pricing is not None,
        })
    return result


def query_summary(days):
    """按用户聚合"""
    detail = query_detail(days)
    users = {}
    for d in detail:
        u = d["user"]
        if u not in users:
            users[u] = {"sessions": 0, "msgs": 0, "input_cache_miss": 0,
                        "cache_read_hit": 0, "output": 0, "cache_write": 0,
                        "total_tokens": 0, "cost_cny": 0}
        for k in ["sessions", "msgs", "input_cache_miss", "cache_read_hit",
                   "output", "cache_write", "total_tokens"]:
            users[u][k] += d[k]
        users[u]["cost_cny"] += d["cost_cny"]

    result = []
    for u, v in sorted(users.items(), key=lambda x: -x[1]["cost_cny"]):
        total_in = v["input_cache_miss"] + v["cache_read_hit"] + v["cache_write"]
        v["cache_hit_rate"] = round(v["cache_read_hit"] / total_in * 100, 1) if total_in > 0 else 0
        v["user"] = u
        v["cost_cny"] = round(v["cost_cny"], 6)
        result.append(v)
    return result


# ── 终端输出 ──────────────────────────────────────────────────
def print_summary(data, days):
    if not data: print("无数据"); return
    total_cny = sum(d["cost_cny"] for d in data)
    print()
    print(f"═══ Token 消耗（近 {days} 天）  总计 {fmt_money_cny(total_cny)} ═══")
    hdr = f"{'用户':10} {'会话':>4} {'←输入(万)':>10} {'缓存命中':>10} {'→输出(万)':>10} {'缓存写':>10} {'总Token':>11} {'费用(¥)':>10} {'命中率':>6}"
    print(hdr)
    print("─" * 100)
    for d in data:
        print(f"{d['user']:10} {d['sessions']:>4} "
              f"{fmt_cn(d['input_cache_miss']):>11} {fmt_cn(d['cache_read_hit']):>10} "
              f"{fmt_cn(d['output']):>11} {fmt_cn(d['cache_write']):>10} "
              f"{fmt_cn(d['total_tokens']):>11} {fmt_money_cny(d['cost_cny']):>10} "
              f"{d['cache_hit_rate']:>5.1f}%")
    print("─" * 100)


def print_detail(data, days):
    if not data: print("无数据"); return
    total_cny = sum(d["cost_cny"] for d in data)
    print()
    print(f"═══ Token 消耗明细（近 {days} 天）  总计 {fmt_money_cny(total_cny)} ═══")
    hdr = f"{'用户':8} {'模型':22} {'会话':>3} {'←输入(万)':>9} {'缓存命中':>8} {'→输出(万)':>8} {'费用(¥)':>9} {'命中率':>6}"
    print(hdr)
    print("─" * 100)
    for d in data:
        inp_m = d["input_cache_miss"] / ONE_MIL
        out_m = d["output"] / ONE_MIL
        print(f"{d['user']:8} {d['model']:22} {d['sessions']:>3} "
              f"{inp_m:>9.3f} {fmt_cn(d['cache_read_hit']):>8} "
              f"{out_m:>8.3f} {fmt_money_cny(d['cost_cny']):>9} "
              f"{d['cache_hit_rate']:>5.1f}%")
    print("─" * 100)
    # 计费说明
    print()
    print("计费公式（¥/百万tokens）：")
    seen = set()
    for d in data:
        key = (d["model"], d["provider"])
        if key in seen: continue
        seen.add(key)
        p = resolve_pricing(d["provider"], d["model"])
        if p:
            parts = [f"{k}=¥{v}" for k, v in p.items()]
            print(f"  {d['model']}: {', '.join(parts)}")
        else:
            print(f"  {d['model']}: ❌ 无定价")


def print_models():
    print()
    print("═══ 已配置模型定价（¥/百万tokens） ═══")
    print(f"{'Provider':15} {'Model':35} {'输入':>8} {'输出':>8} {'缓存读':>8} {'缓存写':>8}")
    print("─" * 90)
    for (bp, model), p in sorted(PRICING.items()):
        vals = [f"¥{p.get(k, '-'):}" for k in ["input", "output", "cache_read", "cache_write"]]
        vals_str = "".join(f"{v:>8}" for v in vals)
        print(f"{bp:15} {model:35}{vals_str}")
    print(f"共 {len(PRICING)} 个模型")

# ── 聚合数据持久化（stats.db）──────────────────────────────────
def _get_stats_conn():
    """获取 stats.db 连接，统一应用安全配置"""
    os.makedirs(os.path.dirname(STATS_DB), exist_ok=True)
    db = sqlite3.connect(STATS_DB)
    db.execute("PRAGMA journal_mode=WAL")          # 崩溃恢复
    db.execute("PRAGMA synchronous=FULL")           # 每次 commit 刷盘
    db.execute("PRAGMA busy_timeout=5000")          # 锁等待 5 秒
    db.execute("PRAGMA foreign_keys=ON")            # 外键约束
    db.execute("PRAGMA wal_autocheckpoint=1000")    # WAL 自动截断
    return db


def _init_stats_db():
    db = _get_stats_conn()
    db.execute("""CREATE TABLE IF NOT EXISTS daily_models (
        date TEXT NOT NULL,
        user TEXT NOT NULL,
        model TEXT NOT NULL,
        provider TEXT,
        sessions INTEGER,
        msgs INTEGER,
        input_cache_miss INTEGER,
        cache_read_hit INTEGER,
        output INTEGER,
        cache_write INTEGER,
        cost_cny REAL,
        PRIMARY KEY (date, user, model)
    )""")
    # 查询加速索引
    db.execute("CREATE INDEX IF NOT EXISTS idx_dm_date ON daily_models(date)")
    db.commit()
    db.close()


def save_daily_stats(data, date_str):
    """将 query_detail 结果按日期存入 stats.db（逐行 upsert）"""
    db = _get_stats_conn()
    try:
        for d in data:
            db.execute("""INSERT OR REPLACE INTO daily_models
                (date, user, model, provider, sessions, msgs,
                 input_cache_miss, cache_read_hit, output, cache_write, cost_cny)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date_str, d['user'], d['model'], d.get('provider', ''),
                 d['sessions'], d['msgs'],
                 d['input_cache_miss'], d['cache_read_hit'], d['output'],
                 d.get('cache_write', 0), d['cost_cny']))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"  ⚠ stats.db 写入失败: {e}")
    finally:
        db.close()

def _catchup():
    """启动时自动补齐所有缺失数据到 stats.db（代替 cron，不生成 HTML）"""
    earliest = None
    for db_path in STATE_DBS:
        try:
            db = sqlite3.connect(db_path)
            e = db.execute("SELECT MIN(started_at) FROM sessions").fetchone()[0]
            db.close()
            if e and (earliest is None or e < earliest):
                earliest = e
        except Exception:
            pass
    if not earliest:
        return  # 静默跳过
    start_date = datetime.fromtimestamp(earliest).date()
    today = date.today()
    count = 0
    d = start_date
    while d < today:
        target_dt = datetime(d.year, d.month, d.day)
        since = target_dt.timestamp()
        until = (target_dt + timedelta(days=1)).timestamp()
        data = query_detail(1, since, until)
        if data:
            save_daily_stats(data, d.strftime('%Y-%m-%d'))
            count += 1
        d += timedelta(days=1)
    # 输出到日志，不输出到终端（后台静默运行）
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] catchup: {count} days collected")


def run_calibrate(filepath):
    """导入 DeepSeek 官方导出的 CSV/ZIP，与 Hermes stats.db 对账。
    支持格式:
    - usage_data_2026_6.zip（含 amount-2026-6.csv）
    - amount-2026-6.csv（直接 CSV）
    
    CSV 列: user_id, utc_date, model, api_key_name, api_key, type, price, amount
    """
    import csv, io, zipfile

    rows = []
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath) as zf:
            for name in zf.namelist():
                if 'amount-' in name and name.endswith('.csv'):
                    with zf.open(name) as f:
                        rows.extend(list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig'))))
    else:
        with open(filepath, encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))

    if not rows:
        print("未找到 amount 数据")
        return

    # 按日期 + key 聚合 token
    # key_map: date -> {key_name: {cache_hit: N, cache_miss: N, output: N}}
    all_days = {}
    key_names = set()
    for r in rows:
        model = r.get('model', '')
        if model != 'deepseek-v4-pro':
            continue
        d = r['utc_date']
        key = r.get('api_key_name', '?')
        typ = r.get('type', '')
        amt = int(r.get('amount', 0))
        key_names.add(key)
        if d not in all_days:
            all_days[d] = {}
        if key not in all_days[d]:
            all_days[d][key] = {'cache_hit': 0, 'cache_miss': 0, 'output': 0}
        if typ == 'input_cache_hit_tokens':
            all_days[d][key]['cache_hit'] += amt
        elif typ == 'input_cache_miss_tokens':
            all_days[d][key]['cache_miss'] += amt
        elif typ == 'output_tokens':
            all_days[d][key]['output'] += amt

    db = sqlite3.connect(STATS_DB)
    print(f"API Keys: {', '.join(sorted(key_names))}")
    print(f"\n{'Date':<12} {'Key':<16} {'Hermes':>14} {'DeepSeek':>14} {'Ratio':>8} {'缺口':>10}")
    print("-" * 80)
    ratios = []
    for d in sorted(all_days.keys()):
        for key in sorted(all_days[d].keys()):
            ds = all_days[d][key]
            ds_total = ds['cache_hit'] + ds['cache_miss'] + ds['output']
            r = db.execute(
                "SELECT SUM(input_cache_miss + cache_read_hit + output) FROM daily_models WHERE date=?",
                (d,)
            ).fetchone()
            h_total = r[0] or 0
            if h_total > 0:
                ratio = ds_total / h_total
                gap = ds_total - h_total
                ratios.append(ratio)
                print(f"{d:<12} {key:<16} {h_total:>14,} {ds_total:>14,} {ratio:>8.4f} {gap:>+10,}")
            else:
                print(f"{d:<12} {key:<16} {'(无数据)':>14} {ds_total:>14,} {'—':>8} {'—':>10}")
    db.close()

    if ratios:
        avg = sum(ratios) / len(ratios)
        std = (sum((r - avg) ** 2 for r in ratios) / len(ratios)) ** 0.5
        print(f"\n{'─' * 80}")
        print(f"统计: {len(ratios)} 条记录（{len(all_days)} 天 × {len(key_names)} keys）")
        print(f"平均校准系数: {avg:.4f}")
        print(f"标准差: {std:.4f}")
        print(f"范围: {min(ratios):.4f} ~ {max(ratios):.4f}")
        if std < 0.15:
            print(f"\n✓ 系数稳定（σ<0.15），可用于费用估算。")
            print(f"  使用方法: Hermes 费用 × {avg:.4f} ≈ DeepSeek 实际费用")
        else:
            print(f"\n⚠ 系数波动较大，建议收集更多数据后再校准。")


def _extract_day(date_str, models, out):
    """从一天的 usage 数据中提取 deepseek-v4-pro 的 token 总量"""
    total = 0
    for m in models:
        if m.get("model") != "deepseek-v4-pro":
            continue
        u = m.get("usage", {})
        if isinstance(u, list):
            # Raw API format: [{type: ..., amount: ...}]
            for entry in u:
                if entry["type"] in ("PROMPT_CACHE_HIT_TOKEN", "PROMPT_CACHE_MISS_TOKEN", "RESPONSE_TOKEN"):
                    total += int(entry["amount"])
        elif isinstance(u, dict):
            # Extension format: {"PROMPT_CACHE_HIT_TOKEN": "amount", ...}
            total += int(u.get("PROMPT_CACHE_HIT_TOKEN", 0))
            total += int(u.get("PROMPT_CACHE_MISS_TOKEN", 0))
            total += int(u.get("RESPONSE_TOKEN", 0))
    if total > 0:
        out[date_str] = total


# ── 本地分词器验证 ──────────────────────────────────────────────
def verify_sessions(days=7):
    """对比本地 tokenizer 计数 vs state.db API 报告的 token 数"""
    t = _get_tokenizer()
    db = sqlite3.connect(DB)
    
    # 获取最近 N 天的会话
    since = (datetime.now() - timedelta(days=days)).timestamp()
    sessions = db.execute("""
        SELECT s.id, s.model, s.input_tokens, s.output_tokens, s.cache_read_tokens,
               s.cache_write_tokens, s.started_at, s.message_count
        FROM sessions s
        WHERE s.model IN ('deepseek-v4-pro', 'deepseek-v4-flash')
          AND s.started_at >= ?
          AND s.input_tokens > 0
        ORDER BY s.started_at DESC
        LIMIT 30
    """, (since,)).fetchall()
    
    if not sessions:
        print("无近期会话数据")
        db.close()
        return
    
    print(f"\n═══ 本地 tokenizer 验证（近 {days} 天，最近 30 个会话） ═══\n")
    print(f"{'会话ID':<22} {'模型':20} {'API输入':>10} {'API输出':>8} {'本地消息':>10} {'API缓存':>10} {'差异':>8}")
    print("-" * 95)
    
    total_api_input = 0
    total_api_output = 0
    total_local = 0
    total_cache = 0
    count = 0
    
    for sid, model, inp, out, cache_r, cache_w, started, msg_count in sessions:
        # 获取该会话所有消息的 content
        messages = db.execute("""
            SELECT COALESCE(content, ''), role FROM messages 
            WHERE session_id = ? AND content IS NOT NULL AND content != ''
            ORDER BY timestamp
        """, (sid,)).fetchall()
        
        if not messages:
            continue
        
        # 拼接所有消息内容
        all_text = '\n'.join(m[0] for m in messages if m[0])
        local_tokens = len(t.encode(all_text).ids)
        
        api_total = (inp or 0) + (out or 0)
        diff = local_tokens - api_total
        diff_pct = (diff / api_total * 100) if api_total > 0 else 0
        
        model_short = model.replace('deepseek-', '')
        print(f"{sid:<22} {model_short:20} {inp or 0:>10,} {out or 0:>8,} {local_tokens:>10,} {(cache_r or 0):>10,} {diff:>+8,}")
        
        total_api_input += (inp or 0)
        total_api_output += (out or 0)
        total_local += local_tokens
        total_cache += (cache_r or 0)
        count += 1
    
    db.close()
    
    api_sum = total_api_input + total_api_output
    print("-" * 95)
    print(f"{'合计':<22} {'':20} {total_api_input:>10,} {total_api_output:>8,} {total_local:>10,} {total_cache:>10,} {total_local - api_sum:>+8,}")
    print(f"\n  API 报告: 输入={total_api_input:,}  输出={total_api_output:,}  缓存命中={total_cache:,}")
    print(f"  本地计数: 消息内容={total_local:,} tokens")
    print(f"  差异: {total_local - api_sum:+,} tokens ({abs(total_local - api_sum)/api_sum*100:.1f}%)" if api_sum > 0 else "")
    print(f"\n  注: 本地计数仅计算消息文本，不含 system prompt、工具输出、缓存效果")
    print(f"      API 报告的 input_tokens 包含完整上下文（system + history + 工具输出）")


def count_mode(text=None):
    """手动测试分词器"""
    if text is None:
        text = sys.argv[2] if len(sys.argv) > 2 else "你好世界"
    
    t = _get_tokenizer()
    tokens = t.encode(text)
    print(f"\n文本: {text}")
    print(f"Token 数: {len(tokens.ids)}")
    print(f"Token IDs: {tokens.ids[:20]}{'...' if len(tokens.ids) > 20 else ''}")
    print(f"解码验证: {t.decode(tokens.ids)}")
    
    # 价格估算
    for model_name, prices in [("v4-flash", {"input": 1, "output": 2}), ("v4-pro", {"input": 3, "output": 6})]:
        cost_in = len(tokens.ids) * prices["input"] / 1_000_000
        cost_out = len(tokens.ids) * prices["output"] / 1_000_000
        print(f"  {model_name}: 输入 ¥{cost_in:.6f}  输出 ¥{cost_out:.6f}")


# ── 入口 ──
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sum"

    if mode in ("collect", "catchup", "calibrate", "models", "count"):
        pass  # 这些模式不需要解析 days
    else:
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    if mode == "models":
        print_models()
    elif mode == "count":
        count_mode()
    elif mode == "verify":
        verify_sessions(days)
    elif mode == "json":
        data = query_detail(days)
        total = sum(d["cost_cny"] for d in data)
        print(json.dumps({
            "days": days, "total_cost_cny": round(total, 6),
            "total_cost_usd": round(total/USD_TO_CNY, 6),
            "exchange_rate": USD_TO_CNY, "detail": data
        }, indent=2, ensure_ascii=False))
    elif mode == "detail":
        print_detail(query_detail(days), days)


    elif mode == "collect":
        # 采集所有日期的数据到 stats.db（不生成报告）
        print("采集数据到 stats.db...")
        earliest = None
        for db_path in STATE_DBS:
            try:
                db = sqlite3.connect(db_path)
                e = db.execute("SELECT MIN(started_at) FROM sessions").fetchone()[0]
                db.close()
                if e and (earliest is None or e < earliest):
                    earliest = e
            except Exception:
                pass
        if not earliest:
            print("无历史数据")
        else:
            start_date = datetime.fromtimestamp(earliest).date()
            today = date.today()
            count = 0
            d = start_date
            while d < today:
                target_dt = datetime(d.year, d.month, d.day)
                since = target_dt.timestamp()
                until = (target_dt + timedelta(days=1)).timestamp()
                data = query_detail(1, since, until)
                if data:
                    save_daily_stats(data, d.strftime('%Y-%m-%d'))
                    count += 1
                    print(f"  ✓ {d}")
                d += timedelta(days=1)
            print(f"✓ 共采集 {count} 天")
    elif mode == "catchup":
        # 启动时自动补齐数据到 stats.db（不生成报告）
        print("📊 Token Stats catchup...")
        _catchup()
    elif mode == "calibrate":
        filepath = sys.argv[2] if len(sys.argv) > 2 else None
        if not filepath or not os.path.exists(filepath):
            print("用法: python3 token_stats.py calibrate <deepseek-usage-export.json>")
            print("      先用浏览器插件导出 JSON，再运行此命令对账")
        else:
            run_calibrate(filepath)
    else:
        try:
            days = int(mode)
        except ValueError:
            pass
        print_summary(query_summary(days), days)