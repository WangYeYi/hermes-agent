"""Tests for FactRetriever FTS5 query sanitization.

These tests cover the fix where raw natural-language queries passed to
FTS5 MATCH were AND-joined by default, dropping recall to zero on any
multi-word prose query. The sanitizer drops stopwords and OR-joins the
remaining content tokens as phrase literals.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


# ---------------------------------------------------------------------------
# _sanitize_fts_query — unit tests (no DB required)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_tokens",
    [
        # stopwords dropped
        ("what happened with the deployment rollback", {"happened", "deployment", "rollback"}),
        # single content word passes through
        ("compaction", {"compaction"}),
        # all stopwords → falls back to raw
        ("the and of", None),  # None = sentinel for fallback-to-raw
        # empty string → empty output
        ("", ""),
        # FTS5 operator characters stripped
        ("context: length-probe", {"context", "lengthprobe"}),
        # trailing punctuation stripped by tokenizer
        ("hello, world!", {"hello", "world"}),
    ],
)
def test_sanitize_fts_query_extracts_content_tokens(query, expected_tokens):
    result = FactRetriever._sanitize_fts_query(query)

    if expected_tokens == "":
        assert result == ""
        return

    if expected_tokens is None:
        # Pathological case: all stopwords — should fall back to raw query
        assert result == query
        return

    # OR-joined phrase literals: `"tok1" OR "tok2" OR ...`
    # Extract the tokens between quotes, order-independent.
    import re
    matches = re.findall(r'"([^"]+)"', result)
    assert set(matches) == expected_tokens, f"got {result!r}"


# ---------------------------------------------------------------------------
# Integration test — actually run _fts_candidates against an in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever_with_facts(tmp_path):
    """MemoryStore seeded with a few facts for retrieval tests."""
    db_path = tmp_path / "test_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="The Thursday deployment rollback failed because of stale migration state.",
        category="project",
    )
    store.add_fact(
        content="Compaction settings tuned to 0.85 threshold.",
        category="tool",
    )
    store.add_fact(
        content="Venice.ai advertises availableContextTokens inside model_spec.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


def test_prefetch_recovers_prose_query(retriever_with_facts):
    """A natural-language query should now match the relevant fact.

    Before the sanitizer fix, 'what happened with the deployment rollback'
    returned zero hits because FTS5 required every token to co-occur.
    """
    results = retriever_with_facts.search(
        "what happened with the deployment rollback"
    )
    assert len(results) >= 1
    # The top hit should be the deployment rollback fact
    assert "deployment rollback" in results[0]["content"].lower()


# ---------------------------------------------------------------------------
# CJK bigram FTS5 tests — zero-recall fix for Chinese / Japanese / Korean
# ---------------------------------------------------------------------------


class TestCjkToFts5Bigrams:
    """Unit tests for _cjk_to_fts5_bigrams — index-side CJK tokenization."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # Pure CJK: 2-gram splits
            ("你好世界", "你好 好世 世界"),
            ("배포는", "배포 포는"),          # Korean 3-char Hangul
            ("こんにちは", "こん んに にち ちは"),  # Japanese Hiragana
            # Mixed script: boundary spaces between ASCII and CJK
            ("Hermes는", "Hermes 는"),        # ASCII + single Hangul
            ("Hello世界", "Hello 世界"),       # ASCII + 2-char CJK
            ("世界Hello", "世界 Hello"),       # CJK + ASCII
            ("A는B", "A 는 B"),               # CJK between ASCII
            # Pure ASCII: unchanged
            ("hello", "hello"),
            ("hello world", "hello world"),
            # Empty / edge
            ("", ""),
            # Single CJK chars
            ("는", "는"),
            ("中", "中"),
        ],
    )
    def test_cjk_bigrams(self, text, expected):
        assert FactRetriever._cjk_to_fts5_bigrams(text) == expected


class TestSanitizeFtsQueryCjk:
    """Unit tests for _sanitize_fts_query — CJK query tokenization."""

    @pytest.mark.parametrize(
        "query,expected_bigrams",
        [
            # Chinese: 2-gram OR expansion
            ("你好", {"你好"}),
            ("你好世界", {"你好", "好世", "世界"}),
            # Korean: Hangul bigrams
            ("배포", {"배포"}),
            ("배포는", {"배포", "포는"}),
            # Japanese
            ("こんにちは", {"こん", "んに", "にち", "ちは"}),
            # Mixed script: CJK + ASCII both present
            ("Hermes 배포", {"hermes", "배포"}),
            # Single CJK char
            ("는", {"는"}),
        ],
    )
    def test_cjk_query_tokens(self, query, expected_bigrams):
        import re as _re
        result = FactRetriever._sanitize_fts_query(query)
        matches = set(_re.findall(r'"([^"]+)"', result))
        assert matches == expected_bigrams, f"got {result!r}"


# ---------------------------------------------------------------------------
# Integration — actual SQLite FTS5 CJK recall
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever_with_cjk_facts(tmp_path):
    """MemoryStore seeded with CJK facts for bigram retrieval tests."""
    db_path = tmp_path / "test_cjk_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="小明每天下午三点去图书馆看书。",
        category="general",
    )
    store.add_fact(
        content="배포 파이프라인이 금요일 오후에 실패했습니다.",
        category="project",
    )
    store.add_fact(
        content="こんにちは、今日の天気はとてもいいです。",
        category="general",
    )
    store.add_fact(
        content="Hermes는 새로운 버전을 배포했습니다.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


@pytest.mark.parametrize(
    "query,expected_substring",
    [
        # Chinese substring recall
        ("图书馆", "图书馆"),
        ("看书", "看书"),
        ("下午三点", "下午三点"),
        # Korean substring recall (the core zero-recall bug)
        ("배포", "배포"),
        ("파이프라인", "파이프라인"),
        ("실패", "실패"),
        # Japanese substring recall
        ("こんにちは", "こんにちは"),
        ("天気", "天気"),
        # Mixed script: search Korean term inside mixed fact
        ("배포", "배포"),  # should match both Korean and mixed facts
    ],
)
def test_cjk_substring_recall(retriever_with_cjk_facts, query, expected_substring):
    """Substring queries against CJK content must return results.

    Before bigram FTS5, FTS5's unicode61 tokenizer treated the entire
    CJK run as one token — any substring query returned zero hits.
    """
    results = retriever_with_cjk_facts.search(query)
    assert len(results) >= 1, f"zero results for CJK query {query!r}"
    contents = [r["content"] for r in results]
    assert any(expected_substring in c for c in contents), (
        f"expected {expected_substring!r} in results for query {query!r}, "
        f"got {contents}"
    )


def test_cjk_fts5_migration_creates_fts_content_column(retriever_with_cjk_facts):
    """After CJK migration, facts table must have an fts_content column."""
    store = retriever_with_cjk_facts.store
    columns = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(facts)").fetchall()
    }
    assert "fts_content" in columns, "fts_content column missing after migration"


def test_cjk_fts5_migration_uses_bigram_index(retriever_with_cjk_facts):
    """FTS5 index must use fts_content column (not old content column)."""
    store = retriever_with_cjk_facts.store
    fts_cols = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(facts_fts)").fetchall()
    }
    assert "fts_content" in fts_cols, "FTS5 not migrated to fts_content"
    assert "content" not in fts_cols, "FTS5 still using old content column"
