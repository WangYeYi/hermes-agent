"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from utils import is_truthy_value
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• resolve — Agent-called: correct fact boosted, wrong fact demoted/deleted.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "resolve", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    try:
        # Canonical loader: behavioral read now honors the managed-scope
        # overlay + ${VAR} expansion (e.g. an api key template) too.
        from hermes_cli.config import load_config_readonly
        all_config = load_config_readonly()
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _replace_entity_value(content: str, entity: str, old_val: str, new_val: str) -> str:
    """Replace an entity's old value with a new value in a fact content string.

    Handles common patterns: '端口:7890', '端口是7890', '端口=7890', '端口 7890'.
    Only replaces the value when the entity name precedes it.
    """
    import re as _re
    pattern = _re.compile(
        _re.escape(entity) + r'\s*[:：=是]?\s*' + _re.escape(old_val),
    )
    replacement = f"{entity}:{new_val}"
    return pattern.sub(replacement, content, count=1)


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            # Write-back round-trip: raw read is correct (merged defaults
            # must not be persisted back into the user's file).
            from hermes_cli.config import read_user_config_raw
            existing = read_user_config_raw(config_path)
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        retrieval_weight = float(self._config.get("retrieval_weight", 0.10))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            retrieval_weight=retrieval_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )

        # Inject high-value facts directly into the system prompt so the agent
        # never has to "remember to retrieve" — the most important facts are
        # always visible. Two cohorts are merged (deduplicated by fact_id):
        #   1. High-trust facts (trust_score >= 0.6) — user-confirmed knowledge
        #   2. Recently added facts — current-session relevant context
        # Max 6 facts total to keep prompt footprint small.
        try:
            facts = self._store._conn.execute(
                "SELECT fact_id, content, trust_score, category FROM facts "
                "WHERE trust_score >= 0.6 AND (deleted IS NULL OR deleted = 0) "
                "ORDER BY updated_at DESC LIMIT 3"
            ).fetchall()
        except Exception:
            facts = []
        seen_ids = {f[0] for f in facts}
        try:
            recent = self._store._conn.execute(
                "SELECT fact_id, content, trust_score, category FROM facts "
                "WHERE fact_id NOT IN ({}) AND (deleted IS NULL OR deleted = 0) "
                "ORDER BY created_at DESC LIMIT 3".format(
                    ",".join(str(rid) for rid in seen_ids) if seen_ids else "0"
                )
            ).fetchall()
        except Exception:
            recent = []
        facts = facts + recent

        header = (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores).\n"
        )
        if not facts:
            return header

        lines = [header, "Injected facts (auto-loaded every turn; no retrieval needed):"]
        for fact in facts:
            tag = "[HIGH]" if fact[2] and fact[2] >= 0.6 else "[RECENT]"
            lines.append(f"- {tag} [{fact[3]}] {fact[1]}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Track fact mentions and corrections in user dialogue.

        Four functions:
        1. _track_mentions(): detect existing facts in user message, boost trust.
        2. _detect_corrections(): detect when user corrects a stored fact, flag
           for resolution (trust -= 0.05 pre-penalty until verified).
        3. _incremental_extract(): when auto_extract is on, harvest seed facts.
        4. resolve_contradiction(): called by agent after verifying correction —
           correct fact boosted, wrong fact demoted or deleted.
        """
        if not self._store or not user_content:
            return
        self._track_mentions(user_content)
        self._detect_corrections(user_content)
        if is_truthy_value(self._config.get("auto_extract", False)):
            self._incremental_extract(user_content)

    def _track_mentions(self, user_content: str) -> int:
        """Detect which existing facts are mentioned in user message, auto-weight.

        Two-layer matching (mutually exclusive, L1 preferred):
        L1: Jaccard token overlap (fast, free, exact keyword matching)
        L2: ONNX semantic similarity (Chinese-optimized, fallback when L1 empty)

        Each hit: trust_score += 0.02 (capped at 1.0), mention_count += 1,
                  last_mentioned_at = NOW, updated_at = NOW.

        Returns number of facts whose trust_score was incremented.
        """
        from .retrieval import FactRetriever

        conn = self._store._conn
        rows = conn.execute(
            "SELECT fact_id, content, trust_score, tags FROM facts "
            "ORDER BY trust_score DESC LIMIT 200"
        ).fetchall()

        if not rows:
            return 0

        user_tokens = FactRetriever._tokenize(user_content)
        if len(user_tokens) < 3:
            return 0  # Too short to meaningfully match

        # L1: Jaccard token overlap
        l1_hits = []
        for row in rows:
            fact_id, content, trust, tags = row
            content_tokens = FactRetriever._tokenize(content)
            tag_tokens = FactRetriever._tokenize(tags or "")
            all_fact_tokens = content_tokens | tag_tokens

            intersection = len(user_tokens & all_fact_tokens)
            union = len(user_tokens | all_fact_tokens)
            jaccard = intersection / union if union > 0 else 0.0

            if jaccard >= 0.15 and intersection >= 3:
                l1_hits.append((fact_id, trust))

        # L2: ONNX semantic (only if L1 found nothing — avoid unnecessary cost)
        if not l1_hits and self._retriever and getattr(self._retriever, '_onnx_available', False):
            try:
                onnx = self._retriever._onnx
                if onnx is None:
                    return 0
                query_vec = onnx.embed(user_content)
                import numpy as np
                query_norm = np.linalg.norm(query_vec)
                if query_norm == 0:
                    return 0
                for row in rows:
                    fact_id, content, trust, tags = row
                    raw = conn.execute(
                        "SELECT onnx_vector FROM facts WHERE fact_id = ?", (fact_id,)
                    ).fetchone()
                    if not raw or not raw[0]:
                        continue
                    fact_vec = np.frombuffer(raw[0], dtype=np.float32)
                    fact_norm = np.linalg.norm(fact_vec)
                    if fact_norm == 0:
                        continue
                    sim = float(np.dot(query_vec, fact_vec) / (query_norm * fact_norm))
                    if sim >= 0.75:
                        l1_hits.append((fact_id, trust))
            except Exception:
                pass

        # Apply weight increments
        updated = 0
        for fact_id, current_trust in l1_hits:
            new_trust = min(1.0, current_trust + 0.02)
            if new_trust > current_trust:
                conn.execute(
                    """UPDATE facts SET
                       trust_score = ?,
                       mention_count = mention_count + 1,
                       last_mentioned_at = CURRENT_TIMESTAMP,
                       updated_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?""",
                    (new_trust, fact_id),
                )
                updated += 1

        if updated:
            conn.commit()

        return updated

    # ------------------------------------------------------------------
    # Correction / contradiction detection
    # ------------------------------------------------------------------

    # Patterns that indicate the user is correcting a previous claim
    _CORRECTION_PATTERNS = [
        # Chinese: "不是X是Y", "X不对，应该是Y", "错了", "正确的是"
        re.compile(r'(?:不是|不对|错了|错误的?).{0,20}(?:是|应该是|正确的是|而是)\s*(.+)'),
        re.compile(r'(.{3,40})\s*(?:不对|错了|不正确|是错的)'),
        re.compile(r'(?:正确的是|应该是|其实是|实际是|真正的?)\s*(.+)'),
        re.compile(r'(?:你又用|你又|你).{0,5}(?:猜测|编造|乱答|瞎猜)'),
        re.compile(r'(?:之前|上次|前面).{0,10}(?:说的?|给的?).{0,10}(?:不对|错了|有问题)'),
        # Direct value flip: "端口是7897" vs stored "端口7890"
        re.compile(r'(?:端口|地址|路径|账号|密码|密钥|key|token|port|host|url)\S{0,3}(?:是|为|改|换|用|设)\s*[:：]?\s*(\S+)'),
        # English
        re.compile(r"(?:that's|that is|you're|you are)\s+(?:wrong|incorrect|mistaken)", re.IGNORECASE),
        re.compile(r"(?:it should be|the correct|actually)\s+(.+)", re.IGNORECASE),
    ]

    # Entity-value patterns for contradiction matching within facts
    # The value capture stops at Chinese/English punctuation or whitespace
    _ENTITY_VALUE_RE = re.compile(
        r'(端口|地址|路径|账号|密码|密钥|key|token|port|host|url|proxy|代理|版本|version|端口号)'
        r'\S{0,3}(?:是|为|:|：|＝|=)\s*([a-zA-Z0-9._-]+)',
        re.IGNORECASE
    )

    def _detect_corrections(self, user_content: str) -> int:
        """Detect user corrections and auto-resolve when the new value is clear.

        Two-tier resolution:
        1. AUTO-RESOLVE: if the user provides a clear entity-value correction
           (e.g. "端口是7897，之前7890错了"), update the fact content with the
           new value immediately. No agent involvement needed.
        2. PRE-PENALTY: if a correction signal is detected but the corrected
           value is ambiguous (token overlap fallback), flag with pre-penalty
           trust -= 0.05 for later resolution.

        Returns number of facts processed (flagged or auto-resolved).
        """
        if not self._store or len(user_content) < 8:
            return 0

        # First: check if any correction pattern matches
        correction_signal = False
        corrected_value = None
        for pat in self._CORRECTION_PATTERNS:
            m = pat.search(user_content)
            if m:
                correction_signal = True
                groups = m.groups()
                if groups and groups[0]:
                    corrected_value = groups[0].strip()
                    if len(corrected_value) > 60:
                        corrected_value = corrected_value[:60]
                break

        if not correction_signal:
            return 0

        conn = self._store._conn
        rows = conn.execute(
            "SELECT fact_id, content, trust_score, tags FROM facts "
            "WHERE trust_score >= 0.3 AND tags NOT LIKE '%correction:resolved%' "
            "ORDER BY trust_score DESC LIMIT 100"
        ).fetchall()

        processed = 0
        auto_resolved = 0
        user_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_-]+', user_content.lower()))

        for row in rows:
            fact_id, content, trust, tags = row
            fact_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_-]+', content.lower()))

            # Extract entity-value pairs from user message and fact
            user_entities = {}
            for m in self._ENTITY_VALUE_RE.finditer(user_content):
                user_entities[m.group(1)] = m.group(2)

            fact_entities = {}
            for m in self._ENTITY_VALUE_RE.finditer(content):
                fact_entities[m.group(1)] = m.group(2)

            is_contradiction = False
            is_agreement = False
            contradicting_entity = None

            for entity, uval in user_entities.items():
                if entity in fact_entities:
                    if fact_entities[entity] != uval:
                        is_contradiction = True
                        contradicting_entity = entity
                    else:
                        is_agreement = True
                    break

            if is_agreement:
                continue

            # Fallback: token overlap without entity-value match
            if not is_contradiction:
                overlap = len(user_tokens & fact_tokens)
                if overlap < 5:
                    continue
                is_contradiction = True

            if not is_contradiction:
                continue

            # ── TIER 1: Auto-resolve when entity-value correction is clear ──
            if contradicting_entity and contradicting_entity in user_entities:
                old_val = fact_entities[contradicting_entity]
                new_val = user_entities[contradicting_entity]

                # Replace old value with corrected value in fact content
                new_content = _replace_entity_value(
                    content, contradicting_entity, old_val, new_val
                )

                new_tags = (tags or "").strip()
                new_tags = new_tags.replace(",correction:pending", "")\
                                   .replace("correction:pending", "")\
                                   .strip(",").strip()
                if "correction:resolved" not in new_tags:
                    new_tags = (new_tags + ",correction:resolved").strip(",")

                conn.execute(
                    "UPDATE facts SET content = ?, tags = ?, trust_score = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                    (new_content, new_tags, trust, fact_id),
                )
                processed += 1
                auto_resolved += 1
                logger.info(
                    "Correction auto-resolved: fact #%d '%s:%s' → '%s:%s'",
                    fact_id, contradicting_entity, old_val,
                    contradicting_entity, new_val,
                )
                continue  # Don't apply pre-penalty; already resolved

            # ── TIER 2: Pre-penalty — ambiguous correction, flag for later ──
            new_trust = max(0.2, trust - 0.05)
            new_tags = (tags or "").strip()
            if "correction:pending" not in new_tags:
                new_tags = (new_tags + ",correction:pending").strip(",")

            conn.execute(
                "UPDATE facts SET trust_score = ?, tags = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                (new_trust, new_tags, fact_id),
            )
            processed += 1
            logger.info(
                "Correction flagged: fact #%d trust %.2f→%.2f (pending verification)",
                fact_id, trust, new_trust,
            )

        if processed:
            conn.commit()

        return processed

    def resolve_contradiction(self, correct_fact_id: int, wrong_fact_id: int) -> dict:
        """Resolve a contradiction after agent verification.

        Called by the agent after the user corrects a fact and the agent verifies
        the correct answer. This is NOT manual fact_feedback — it's triggered
        automatically by the agent's verification workflow.

        correct_fact_id: the fact that was verified as correct (boost trust).
        wrong_fact_id: the fact that was wrong (demote or delete).

        Returns dict with resolution details.
        """
        if not self._store:
            return {"status": "error", "reason": "store not available"}

        conn = self._store._conn
        result = {"status": "ok", "corrected": None, "demoted": None, "deleted": None}

        # Boost the correct fact
        correct_row = conn.execute(
            "SELECT fact_id, content, trust_score, tags FROM facts WHERE fact_id = ?",
            (correct_fact_id,),
        ).fetchone()

        if correct_row:
            new_trust = min(1.0, correct_row[2] + 0.05)
            new_tags = correct_row[3] or ""
            new_tags = new_tags.replace(",correction:pending", "").replace("correction:pending", "")
            conn.execute(
                "UPDATE facts SET trust_score = ?, tags = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE fact_id = ?",
                (new_trust, new_tags.strip(",").strip(), correct_fact_id),
            )
            result["corrected"] = {
                "fact_id": correct_fact_id,
                "old_trust": correct_row[2],
                "new_trust": new_trust,
                "content": correct_row[1][:80],
            }
            logger.info(
                "Contradiction resolved: correct fact #%d trust %.2f→%.2f",
                correct_fact_id, correct_row[2], new_trust,
            )

        # Demote or delete the wrong fact
        wrong_row = conn.execute(
            "SELECT fact_id, content, trust_score, tags FROM facts WHERE fact_id = ?",
            (wrong_fact_id,),
        ).fetchone()

        if wrong_row:
            old_trust = wrong_row[2]
            new_trust = old_trust - 0.15  # Significant penalty for being wrong

            if new_trust < 0.2:
                # Delete: too low to be useful
                conn.execute("DELETE FROM facts WHERE fact_id = ?", (wrong_fact_id,))
                result["deleted"] = {
                    "fact_id": wrong_fact_id,
                    "old_trust": old_trust,
                    "content": wrong_row[1][:80],
                }
                logger.info(
                    "Contradiction resolved: wrong fact #%d deleted (trust %.2f→below floor)",
                    wrong_fact_id, old_trust,
                )
            else:
                new_tags = (wrong_row[3] or "") + ",correction:demoted"
                new_tags = new_tags.strip(",").replace("correction:pending,", "").replace(",correction:pending", "")
                conn.execute(
                    "UPDATE facts SET trust_score = ?, tags = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE fact_id = ?",
                    (new_trust, new_tags, wrong_fact_id),
                )
                result["demoted"] = {
                    "fact_id": wrong_fact_id,
                    "old_trust": old_trust,
                    "new_trust": new_trust,
                    "content": wrong_row[1][:80],
                }
                logger.info(
                    "Contradiction resolved: wrong fact #%d trust %.2f→%.2f",
                    wrong_fact_id, old_trust, new_trust,
                )

        conn.commit()
        return result

    def _incremental_extract(self, user_content: str) -> None:
        """Extract seed facts from a single user message (Chinese + English patterns).

        Called from sync_turn() when auto_extract is enabled. Uses the same
        patterns as _auto_extract_facts() but operates on individual messages
        instead of the full session transcript. Extracted facts get default_trust
        (0.5); the mention-tracking system (change #2) will automatically boost
        trust_score for facts the user repeatedly references in later turns.
        """
        if not self._store or len(user_content) < 15:
            return

        # Preference patterns (user_pref category)
        _CN_EN_PREF = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
            re.compile(r'(?:我|阿锋).{0,10}(?:喜欢|习惯|一般|通常|总是|从来|基本).{2,30}(?:用|做|选|设|是)'),
            re.compile(r'(?:我|阿锋).{0,5}(?:偏好|倾向|首选|默认)'),
            re.compile(r'(?:我的|我们的).{2,20}(?:是|用|设|选|放在)'),
            re.compile(r'(?:以后|接下来|从现在起).{2,30}(?:都|要|用|按)'),
            re.compile(r'(?:禁止|不允许|不要|别).{0,10}(?:用|做|设|改)'),
        ]

        # Decision patterns (project category)
        _CN_EN_DECISION = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
            re.compile(r'(?:我们|方案).{0,10}(?:决定|确定|选择|采用)'),
            re.compile(r'(?:项目|系统|代码).{0,5}(?:使用|采用|依赖|基于)'),
            re.compile(r'(?:最终|结论|所以).{0,10}(?:方案|做法|方式)'),
            re.compile(r'(?:已|已经|前面).{0,5}(?:确认|验证|测试).{0,10}(?:通过|没问题|正常)'),
        ]

        for pattern in _CN_EN_PREF:
            if pattern.search(user_content):
                try:
                    self._store.add_fact(user_content[:400], category="user_pref")
                except Exception:
                    pass
                return  # One fact per message max

        for pattern in _CN_EN_DECISION:
            if pattern.search(user_content):
                try:
                    self._store.add_fact(user_content[:400], category="project")
                except Exception:
                    pass
                return

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Always apply temporal decay: stale facts lose trust over time.
        self._apply_temporal_decay()

        # Auto-resolve any pending corrections that were flagged but not cleared
        if self._store:
            self._auto_resolve_pending_corrections()

        if not is_truthy_value(self._config.get("auto_extract", False)):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def _auto_resolve_pending_corrections(self) -> int:
        """Auto-resolve facts still tagged correction:pending.

        For each pending fact, search for another fact sharing the same entity
        but with a different value — this is the "correct" version. If found,
        auto-resolve: boost the correct one (+0.05), demote the pending one
        (-0.15 or delete if trust < 0.2). This cleans up corrections that
        weren't resolved immediately because the entity-value match was
        ambiguous at detection time.
        """
        if not self._store:
            return 0

        conn = self._store._conn
        pending = conn.execute(
            "SELECT fact_id, content, trust_score, tags FROM facts "
            "WHERE tags LIKE '%correction:pending%'"
        ).fetchall()

        if not pending:
            return 0

        resolved = 0
        for p_row in pending:
            p_id, p_content, p_trust, p_tags = p_row

            # Extract entities from this pending fact
            p_entities = {}
            for m in self._ENTITY_VALUE_RE.finditer(p_content):
                p_entities[m.group(1)] = m.group(2)

            if not p_entities:
                continue

            # Search for a "correct" version: same entity, different value, no pending tag
            for entity, p_val in p_entities.items():
                candidates = conn.execute(
                    "SELECT fact_id, content, trust_score, tags FROM facts "
                    "WHERE fact_id != ? AND content LIKE ? "
                    "AND tags NOT LIKE '%correction:pending%' "
                    "ORDER BY trust_score DESC LIMIT 5",
                    (p_id, f"%{entity}%"),
                ).fetchall()

                for c_row in candidates:
                    c_id, c_content, c_trust, c_tags = c_row
                    c_entities = {}
                    for m in self._ENTITY_VALUE_RE.finditer(c_content):
                        c_entities[m.group(1)] = m.group(2)

                    if entity in c_entities and c_entities[entity] != p_val:
                        # Found a contradicting pair → auto-resolve
                        self.resolve_contradiction(c_id, p_id)
                        resolved += 1
                        break  # One resolution per pending fact

                if resolved > 0:
                    break  # Already resolved this pending fact

        if resolved:
            conn.commit()

        return resolved

    def _apply_temporal_decay(self, days_threshold: int = 60) -> int:
        """Reduce trust_score for facts not updated in `days_threshold` days.

        Each decay tick: trust_score -= 0.03 (floor: 0.2). Facts below 0.3
        are never touched — they're already in the "low-confidence" zone.
        A fact with last_mentioned_at or updated_at within the threshold is
        spared (the user or system interacted with it recently).

        Returns number of facts decayed.
        """
        if not self._store:
            return 0

        conn = self._store._conn
        rows = conn.execute(
            """SELECT fact_id, trust_score FROM facts
               WHERE trust_score > 0.3
                 AND (last_mentioned_at IS NULL OR last_mentioned_at < datetime('now', ?))
                 AND updated_at < datetime('now', ?)""",
            (f"-{days_threshold} days", f"-{days_threshold} days"),
        ).fetchall()

        updated = 0
        for row in rows:
            fact_id, trust = row
            new_trust = max(0.2, trust - 0.03)
            if new_trust < trust:
                conn.execute(
                    "UPDATE facts SET trust_score = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE fact_id = ?",
                    (new_trust, fact_id),
                )
                updated += 1

        if updated:
            conn.commit()
            logger.info("Decayed %d stale facts (trust -= 0.03)", updated)

        return updated

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        # Release the shared SQLite connection deterministically on the
        # caller's thread. Dropping the reference alone leaves fd finalization
        # to GC, which keeps the connection (and its write lock) alive on a
        # long-running gateway and prolongs the "database is locked" contention
        # this store's shared-connection refcounting is meant to eliminate.
        # close() is idempotent and refcount-guarded, so siblings stay safe.
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "resolve":
                correct_id = int(args.get("correct_fact_id", 0))
                wrong_id = int(args.get("wrong_fact_id", 0))
                if not correct_id or not wrong_id:
                    return tool_error("resolve requires 'correct_fact_id' and 'wrong_fact_id'")
                result = self.resolve_contradiction(correct_id, wrong_id)
                return json.dumps(result)

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        # Local import (pattern used in initialize()): the compressor module is
        # heavier than this plugin and is only needed when auto_extract is on.
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            is_compaction_summary_message,
        )

        def _pre_delimiter_user_segment(msg: dict):
            """Return the genuine user text preceding a merged-into-tail
            compaction summary, or None when the whole message is a summary.

            Merge-into-tail messages (agent/context_compressor.py ~3163-3190)
            wrap real prior tail content BEFORE ``_MERGED_SUMMARY_DELIMITER``,
            prefixed with ``_MERGED_PRIOR_CONTEXT_HEADER``, then append the
            generated handoff summary AFTER the delimiter. Dropping the whole
            row (as ``is_compaction_summary_message`` alone would suggest)
            discards that genuine pre-delimiter content too (#57690 review).
            Only the summary suffix must be excluded from harvesting.
            """
            content = msg.get("content", "")
            if not isinstance(content, str) or _MERGED_SUMMARY_DELIMITER not in content:
                return None
            pre = content.split(_MERGED_SUMMARY_DELIMITER, 1)[0]
            if pre.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
                pre = pre[len(_MERGED_PRIOR_CONTEXT_HEADER):]
            pre = pre.strip()
            return pre or None

        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
            # Chinese preference / personal habit patterns
            re.compile(r'(?:我|阿锋).{0,10}(?:喜欢|习惯|一般|通常|总是|从来|基本).{2,30}(?:用|做|选|设|是)'),
            re.compile(r'(?:我|阿锋).{0,5}(?:偏好|倾向|首选|默认)'),
            re.compile(r'(?:我的|我们的).{2,20}(?:是|用|设|选|放在)'),
            re.compile(r'(?:以后|接下来|从现在起).{2,30}(?:都|要|用|按)'),
            re.compile(r'(?:禁止|不允许|不要|别).{0,10}(?:用|做|设|改)'),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
            # Chinese decision / project patterns
            re.compile(r'(?:我们|方案).{0,10}(?:决定|确定|选择|采用)'),
            re.compile(r'(?:项目|系统|代码).{0,5}(?:使用|采用|依赖|基于)'),
            re.compile(r'(?:最终|结论|所以).{0,10}(?:方案|做法|方式)'),
            re.compile(r'(?:已|已经|前面).{0,5}(?:确认|验证|测试).{0,10}(?:通过|没问题|正常)'),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            # Compaction handoff summaries can be inserted as role="user"
            # messages; their prose reliably matches the decision patterns, so
            # without this guard the compactor's own output is stored as a
            # durable "fact" on every rollover (#57682). A merge-into-tail
            # summary also carries genuine pre-delimiter user content in the
            # SAME row; harvest that segment instead of dropping the whole
            # message (#57690 review).
            pre_delimiter_segment = _pre_delimiter_user_segment(msg)
            if pre_delimiter_segment is not None:
                content = pre_delimiter_segment
            elif is_compaction_summary_message(msg):
                continue
            else:
                content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
