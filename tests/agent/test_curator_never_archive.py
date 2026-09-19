"""Local patch: curator.never_archive exempts named skills from automatic transitions.

Why this exists: a bundled built-in can be pruned (curator.prune_builtins) yet can NEVER be
pinned — `hermes curator pin` refuses bundled/hub-installed skills — so there was no
config-side way to keep a shipped skill out of the deterministic pass. This covers the
exemption list: no stale, no archive, empty list = unchanged behaviour.
"""
from datetime import datetime, timedelta, timezone

import pytest

import agent.curator as curator
from tools import skill_usage as usage

OLD = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()


def _row(name, **kw):
    row = {"name": name, "state": "active", "use_count": 5, "last_activity_at": OLD,
           "_persisted": True, "pinned": False}
    row.update(kw)
    return row


@pytest.fixture
def wired(monkeypatch):
    """Record what the pass would do, without touching disk."""
    calls = {"archived": [], "states": []}
    monkeypatch.setattr(curator, "_cron_referenced_skills", lambda: set())
    monkeypatch.setattr(usage, "seed_record_if_missing", lambda name: None)
    monkeypatch.setattr(usage, "set_state",
                        lambda name, state, **kw: calls["states"].append((name, state)))
    monkeypatch.setattr(curator, "_archive_as_curator",
                        lambda u, name: (calls["archived"].append(name), True)[1])
    return calls


def _run(monkeypatch, names, never_archive):
    monkeypatch.setattr(usage, "curated_report", lambda: [_row(n) for n in names])
    monkeypatch.setattr(curator, "_load_config",
                        lambda: ({"never_archive": never_archive} if never_archive is not None else {}))
    return curator.apply_automatic_transitions()


def test_exempt_skill_is_not_archived(monkeypatch, wired):
    counts = _run(monkeypatch, ["keepme", "droppy"], ["keepme"])
    assert wired["archived"] == ["droppy"]
    assert counts["archived"] == 1


def test_exempt_skill_is_not_marked_stale(monkeypatch, wired):
    """跳过 = 连 stale 标记都不打（列表里状态不该有变化）。"""
    young = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    monkeypatch.setattr(usage, "curated_report",
                        lambda: [_row("keepme", last_activity_at=young), _row("other", last_activity_at=young)])
    monkeypatch.setattr(curator, "_load_config", lambda: {"never_archive": ["keepme"]})
    curator.apply_automatic_transitions()
    assert wired["states"] == [("other", "stale")]


def test_empty_list_changes_nothing(monkeypatch, wired):
    counts = _run(monkeypatch, ["a", "b"], [])
    assert sorted(wired["archived"]) == ["a", "b"]
    assert counts["archived"] == 2


def test_missing_key_changes_nothing(monkeypatch, wired):
    _run(monkeypatch, ["a"], None)
    assert wired["archived"] == ["a"]


def test_single_string_and_blank_entries_tolerated(monkeypatch, wired):
    _run(monkeypatch, ["keepme", "droppy"], [" keepme ", ""])
    assert wired["archived"] == ["droppy"]


def test_mutation_control_exempt_name_is_archived_without_the_list(monkeypatch, wired):
    """变异对照：名单消失后同一技能必须被归档 —— 证明上面的断言有判别力。"""
    _run(monkeypatch, ["keepme"], None)
    assert wired["archived"] == ["keepme"]
