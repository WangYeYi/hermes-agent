"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest




def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh and HEAD is unchanged, return the cached value without fetching.

    A fresh cache for a source install still runs one local `git rev-parse HEAD`
    (to confirm HEAD hasn't moved, see #40944) but must NOT run the network
    `git fetch` / `git rev-list` recheck.
    """
    from hermes_cli.banner import check_for_updates
    from hermes_cli import __version__

    # Create a fake git repo and fresh cache stamped with the current HEAD.
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps(
        {"ts": time.time(), "behind": 3, "rev": None, "ver": __version__, "head": "cafef00d"}
    ))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    # `git rev-parse HEAD` reports the same hash the cache was stamped with.
    mock_result = MagicMock(returncode=0, stdout="cafef00d\n")
    with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = check_for_updates()

    assert result == 3
    assert mock_run.call_count == 1  # only rev-parse HEAD, no fetch/rev-list


def test_check_for_updates_invalidates_on_head_change(tmp_path, monkeypatch):
    """A fresh cache from a different local HEAD must be re-checked, not reused.

    Regression for #40944: after a manual `git pull --ff-only` on a source
    install, HEAD moves but VERSION and the embedded rev are unchanged, so the
    6h-TTL cache kept reporting the stale 'behind' count. The HEAD guard forces
    a recheck, which now reports 0 commits behind.
    """
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    # Point _resolve_repo_dir() at our fake checkout (its __file__ preference
    # would otherwise resolve to the real repo running the tests).
    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Fresh (within TTL) cache that says "behind 81", stamped with the OLD HEAD.
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps(
        {"ts": time.time(), "behind": 81, "rev": None, "ver": banner.VERSION, "head": "0ldc0mmit"}
    ))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    def fake_run(cmd, *args, **kwargs):
        # rev-parse reports the NEW post-pull HEAD; rev-list reports 0 behind.
        if "rev-parse" in cmd:
            return MagicMock(returncode=0, stdout="new00000\n")
        if "rev-list" in cmd:
            return MagicMock(returncode=0, stdout="0\n")
        return MagicMock(returncode=0, stdout="")  # git fetch

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner.check_for_updates()

    # Stale-HEAD cache rejected -> fresh check ran -> up-to-date result.
    assert result == 0
    written = json.loads(cache_file.read_text())
    assert written["behind"] == 0
    assert written["head"] == "new00000"






def test_prefetch_non_blocking():
    """prefetch_update_check() should return immediately without blocking."""
    import hermes_cli.banner as banner

    # Reset module state
    banner._update_result = None
    banner._update_check_done = threading.Event()

    with patch.object(banner, "check_for_updates", return_value=5):
        start = time.monotonic()
        banner.prefetch_update_check()
        elapsed = time.monotonic() - start

        # Should return almost immediately (well under 1 second)
        assert elapsed < 1.0

        # Wait for the background thread to finish
        banner._update_check_done.wait(timeout=5)
        assert banner._update_result == 5


def test_check_via_local_git_fetch_failure_returns_none(tmp_path, monkeypatch):
    """When git fetch fails and the stale origin/main ref is not ahead,
    _check_via_local_git must return None (#82166).

    A stale tracking ref cannot prove *currentness* (rev-list 0 just means
    the ref hasn't caught up), so returning None is the honest inconclusive
    result — and the caller must not cache it as "up to date".
    """
    from hermes_cli import banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Simulate a non-shallow, non-SSH-remote checkout
    def mock_git_stdout(args, *, cwd, timeout=5):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args[:2] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return None

    # Fetch fails (returncode != 0); stale rev-list reports 0 behind
    failed_proc = MagicMock()
    failed_proc.returncode = 1
    failed_proc.stdout = ""
    failed_proc.stderr = "fatal: could not reach remote"

    stale_zero_proc = MagicMock()
    stale_zero_proc.returncode = 0
    stale_zero_proc.stdout = "0"

    def mock_run(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            return failed_proc
        if args[:2] == ["git", "rev-list"]:
            return stale_zero_proc
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(banner, "_git_stdout", mock_git_stdout)
    monkeypatch.setattr(banner.subprocess, "run", mock_run)

    result = banner._check_via_local_git(repo_dir)
    assert result is None, (
        "Fetch failure with stale 0-behind must return None, not 'up to date'"
    )


def test_check_via_local_git_fetch_failure_keeps_positive_stale_count(tmp_path, monkeypatch):
    """A failed fetch must preserve sound evidence: if the stale origin/main
    ref already shows HEAD behind, that positive count is still an update
    signal and must be returned (review #92578)."""
    from hermes_cli import banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def mock_git_stdout(args, *, cwd, timeout=5):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args[:2] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return None

    failed_proc = MagicMock()
    failed_proc.returncode = 1
    failed_proc.stdout = ""
    failed_proc.stderr = "fatal: could not reach remote"

    stale_behind_proc = MagicMock()
    stale_behind_proc.returncode = 0
    stale_behind_proc.stdout = "5"

    def mock_run(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            return failed_proc
        if args[:2] == ["git", "rev-list"]:
            return stale_behind_proc
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(banner, "_git_stdout", mock_git_stdout)
    monkeypatch.setattr(banner.subprocess, "run", mock_run)

    result = banner._check_via_local_git(repo_dir)
    assert result == 5, "Stale positive behind-count must be preserved on fetch failure"


def test_check_via_local_git_fetch_failure_rev_list_error_returns_none(tmp_path, monkeypatch):
    """If the stale rev-list itself fails, the check stays inconclusive (None)."""
    from hermes_cli import banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def mock_git_stdout(args, *, cwd, timeout=5):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args[:2] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return None

    failed_proc = MagicMock()
    failed_proc.returncode = 1
    failed_proc.stdout = ""
    failed_proc.stderr = "fatal: could not reach remote"

    bad_rev_list = MagicMock()
    bad_rev_list.returncode = 128
    bad_rev_list.stdout = ""
    bad_rev_list.stderr = "fatal: ambiguous argument 'HEAD..origin/main'"

    def mock_run(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            return failed_proc
        if args[:2] == ["git", "rev-list"]:
            return bad_rev_list
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(banner, "_git_stdout", mock_git_stdout)
    monkeypatch.setattr(banner.subprocess, "run", mock_run)

    result = banner._check_via_local_git(repo_dir)
    assert result is None


def test_check_for_updates_does_not_cache_none(tmp_path, monkeypatch):
    """check_for_updates must not cache None results so a transient fetch
    failure doesn't suppress retries for the full 6-hour cache window (#82166).

    Instead of mocking the full Path resolution chain, we verify the cache-write
    guard directly: call check_for_updates with a mocked _check_via_local_git
    that returns None, and confirm no cache file is created.
    """
    import hermes_cli.banner as banner

    cache_file = tmp_path / ".update_check"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    # Create a fake repo dir so the .git check passes
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Mock the internal functions to force the local-git path returning None
    monkeypatch.setattr(banner, "_check_via_local_git", lambda rd: None)
    monkeypatch.setattr(
        "hermes_cli.config.detect_install_method", lambda root: "git"
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_project_root", lambda: repo_dir
    )

    # Patch __file__ resolution by monkeypatching the module's Path calls.
    # check_for_updates does: Path(__file__).parent.parent.resolve()
    # We intercept by making the resolve() return our fake repo_dir.
    original_init = Path.__init__

    def patched_path_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

    # Simpler: just patch the get_hermes_home and the repo_dir resolution
    # by making check_for_updates find our fake repo via hermes_home fallback.
    # The code checks Path(__file__).parent.parent/.git first, then falls
    # back to hermes_home / "hermes-agent". We ensure the fallback hits.
    # To do this, we make Path(__file__).parent.parent.resolve() return
    # a path without .git, so it falls through to hermes_home / "hermes-agent".
    real_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        s = str(self)
        if "banner.py" in s or s.endswith("hermes_cli"):
            # Return a path that has no .git, forcing the fallback
            return tmp_path / "no-git-here"
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    result = banner.check_for_updates()
    assert result is None

    # The cache file must NOT have been written with a None result
    assert not cache_file.exists(), "None result must not be cached"




