"""Dependency pin upgrades must preserve local edits in cached repositories."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from elmm_pipeline import bootstrap
from elmm_pipeline.bootstrap import ensure_repo


class RepositoryPinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.git(self.upstream, "init", "--quiet")
        self.first = self.commit("first")
        self.spec = {"url": str(self.upstream), "revision": self.first}
        self.patch = patch("elmm_pipeline.bootstrap.ROOT", self.root / "pipeline")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cache = ensure_repo("example", self.spec)

    def git(self, directory, *arguments):
        return subprocess.run(["git", *arguments], cwd=directory, capture_output=True,
                              text=True, check=True).stdout.strip()

    def commit(self, value):
        (self.upstream / "source.f90").write_text(value + "\n")
        self.git(self.upstream, "add", "source.f90")
        self.git(self.upstream, "-c", "user.name=Dependency test", "-c", "user.email=dependency-test@example.invalid",
                 "commit", "--quiet", "-m", value)
        return self.git(self.upstream, "rev-parse", "HEAD")

    def test_unchanged_pin_reuses_cache_without_fetch(self):
        with patch("elmm_pipeline.bootstrap.command", wraps=bootstrap.command) as command:
            self.assertEqual(ensure_repo("example", self.spec), self.cache)
        self.assertFalse(any(call.args[0][1] == "fetch" for call in command.call_args_list))

    def test_new_pin_fetches_and_checks_out_clean_cache(self):
        second = self.commit("second")
        result = ensure_repo("example", dict(self.spec, revision=second))
        self.assertEqual(result, self.cache)
        self.assertEqual(self.git(self.cache, "rev-parse", "HEAD"), second)
        self.assertEqual((self.cache / "source.f90").read_text(), "second\n")
        self.assertEqual(self.git(self.cache, "status", "--porcelain"), "")

    def test_tracked_edits_survive_pin_update_refusal(self):
        second = self.commit("second")
        (self.cache / "source.f90").write_text("local edit\n")
        with self.assertRaisesRegex(RuntimeError, "source changes"):
            ensure_repo("example", dict(self.spec, revision=second))
        self.assertEqual((self.cache / "source.f90").read_text(), "local edit\n")
        self.assertEqual(self.git(self.cache, "rev-parse", "HEAD"), self.first)

    def test_untracked_files_survive_pin_update_refusal(self):
        second = self.commit("second")
        note = self.cache / "local-notes.txt"
        note.write_text("preserve me\n")
        with self.assertRaisesRegex(RuntimeError, "source changes"):
            ensure_repo("example", dict(self.spec, revision=second))
        self.assertEqual(note.read_text(), "preserve me\n")
        self.assertEqual(self.git(self.cache, "rev-parse", "HEAD"), self.first)

    def test_ignored_local_file_is_not_overwritten_by_new_pin(self):
        (self.cache / ".git/info/exclude").write_text("local.txt\n")
        (self.cache / "local.txt").write_text("local ignored content\n")
        (self.upstream / "local.txt").write_text("new upstream content\n")
        self.git(self.upstream, "add", "local.txt")
        second = self.commit("second")
        with self.assertRaises(RuntimeError):
            ensure_repo("example", dict(self.spec, revision=second))
        self.assertEqual((self.cache / "local.txt").read_text(), "local ignored content\n")
        self.assertEqual(self.git(self.cache, "rev-parse", "HEAD"), self.first)


if __name__ == "__main__":
    unittest.main()
