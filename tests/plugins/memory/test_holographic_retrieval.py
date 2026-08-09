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


# ---------------------------------------------------------------------------
# hrr_dim mismatch across sessions — issue #68682
# ---------------------------------------------------------------------------

@pytest.fixture
def mismatched_dim_retriever(tmp_path):
    """Facts encoded at hrr_dim=256, then queried by a retriever configured
    for hrr_dim=1024 — simulating a config change between sessions."""
    db_path = tmp_path / "mismatch.db"
    store = MemoryStore(str(db_path), hrr_dim=256)
    store.add_fact("Peppi works on the backend team.", category="project")
    store.add_fact("The backend uses Postgres.", category="project")
    retriever = FactRetriever(store=store, hrr_dim=1024)
    yield retriever, store
    store.close()


class TestDimMismatchDoesNotCrash:
    def test_search_returns_without_raising(self, mismatched_dim_retriever):
        retriever, _store = mismatched_dim_retriever
        results = retriever.search("backend")
        assert isinstance(results, list)

    def test_probe_returns_without_raising(self, mismatched_dim_retriever):
        retriever, _store = mismatched_dim_retriever
        results = retriever.probe("peppi", category="project")
        assert isinstance(results, list)

    def test_related_returns_without_raising(self, mismatched_dim_retriever):
        retriever, _store = mismatched_dim_retriever
        results = retriever.related("peppi", category="project")
        assert isinstance(results, list)

    def test_reason_returns_without_raising(self, mismatched_dim_retriever):
        retriever, _store = mismatched_dim_retriever
        results = retriever.reason(["peppi", "backend"], category="project")
        assert isinstance(results, list)

    def test_contradict_returns_without_raising(self, mismatched_dim_retriever):
        retriever, _store = mismatched_dim_retriever
        results = retriever.contradict(category="project")
        assert isinstance(results, list)

    def test_probe_all_rows_skipped_returns_empty_and_warns_once(self, tmp_path, caplog):
        """When every fact vector in a category mismatches the retriever's
        dim (e.g. a whole category migrated at a different time), probe()
        must return [] (not fall through to a keyword-search that finds
        stale/irrelevant hits) and log exactly one warning naming
        rebuild_all_vectors."""
        import logging

        db_path = tmp_path / "all_skip.db"
        store = MemoryStore(str(db_path), hrr_dim=256)
        store.add_fact("Peppi owns the deploy pipeline.", category="project")
        retriever = FactRetriever(store=store, hrr_dim=1024)
        try:
            with caplog.at_level(logging.WARNING, logger="plugins.memory.holographic.retrieval"):
                results = retriever.probe("peppi", category="project")
            assert results == []
            warnings = [r for r in caplog.records if "skipped" in r.message.lower()]
            assert len(warnings) == 1
            assert "rebuild_all_vectors" in warnings[0].message
        finally:
            store.close()

    def test_probe_corrupt_bank_with_no_fact_vectors_still_warns(self, tmp_path, caplog):
        """A mismatched bank vector counted as skipped must not vanish
        silently through probe()'s no-fact-rows early return (keyword
        fallback) — the operator still needs the rebuild_all_vectors hint."""
        import logging

        db_path = tmp_path / "bank_only.db"
        store = MemoryStore(str(db_path), hrr_dim=256)
        store.add_fact("Peppi owns the deploy pipeline.", category="project")
        # Strip the per-fact vectors so only the (mismatched) bank remains.
        store._conn.execute("UPDATE facts SET hrr_vector = NULL")
        store._conn.commit()
        retriever = FactRetriever(store=store, hrr_dim=1024)
        try:
            with caplog.at_level(logging.WARNING, logger="plugins.memory.holographic.retrieval"):
                retriever.probe("peppi", category="project")
            warnings = [r for r in caplog.records if "skipped" in r.message.lower()]
            assert len(warnings) == 1
            assert "rebuild_all_vectors" in warnings[0].message
        finally:
            store.close()

    def test_skip_warning_logged_once_per_operation(self, mismatched_dim_retriever, caplog):
        import logging
        from plugins.memory.holographic import retrieval as retrieval_mod

        retriever, _store = mismatched_dim_retriever
        with caplog.at_level(logging.WARNING, logger=retrieval_mod.__name__):
            retriever.related("peppi", category="project")

        skip_warnings = [
            r for r in caplog.records if "skipped" in r.message.lower()
        ]
        assert len(skip_warnings) == 1
        assert "rebuild_all_vectors" in skip_warnings[0].message


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
