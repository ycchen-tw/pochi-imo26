import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relocate_runtime import relocate_runtime


class RuntimeRelocationTests(unittest.TestCase):
    def test_rewrites_old_nfs_and_container_entrypoints_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            scripts = runtime / "venv/bin"
            scripts.mkdir(parents=True)
            (runtime / "pybase/bin").mkdir(parents=True)
            (runtime / "pybase/bin/python3").write_bytes(b"python")
            (runtime / "venv/pyvenv.cfg").write_text(
                "home = /nfs/aimo/shared/fm-pochi/runtime/pybase/bin\n"
                "version_info = 3.12.13\n"
            )
            tool = scripts / "tool"
            tool.write_bytes(
                b"#!/nfs/aimo/shared/fm-pochi/runtime/venv/bin/python3\n"
                b"print('ok')\n"
            )
            tool.chmod(0o755)
            link = scripts / "python-link"
            link.symlink_to("/opt/pp/pybase/bin/python3")

            first = relocate_runtime(runtime)
            self.assertTrue(first["pyvenv_cfg_changed"])
            self.assertEqual(first["rewritten_scripts"], ["tool"])
            self.assertEqual(first["rewritten_symlinks"], ["python-link"])
            self.assertIn(str(runtime / "pybase/bin"), (runtime / "venv/pyvenv.cfg").read_text())
            self.assertEqual(
                tool.read_bytes().splitlines()[0],
                f"#!{runtime / 'venv/bin/python3'}".encode(),
            )
            self.assertFalse(os.readlink(link).startswith("/opt/pp/"))

            second = relocate_runtime(runtime)
            self.assertFalse(second["pyvenv_cfg_changed"])
            self.assertEqual(second["rewritten_scripts"], [])
            self.assertEqual(second["rewritten_symlinks"], [])


if __name__ == "__main__":
    unittest.main()
