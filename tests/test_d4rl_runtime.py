import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.d4rl_runtime import (
    EGL_VENDOR_ENV,
    configure_nvidia_egl_vendor,
)


class NvidiaEglVendorTest(unittest.TestCase):
    def test_preserves_an_explicit_vendor_setting(self):
        with mock.patch.dict(os.environ, {EGL_VENDOR_ENV: "/custom/vendor.json"}):
            selected = configure_nvidia_egl_vendor(Path("/unused"))

            self.assertEqual(selected, Path("/custom/vendor.json"))
            self.assertEqual(os.environ[EGL_VENDOR_ENV], "/custom/vendor.json")

    def test_uses_bundled_vendor_when_nvidia_egl_library_is_available(self):
        with TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            vendor_file = repo_root / "configs/nvidia_egl_vendor.json"
            vendor_file.parent.mkdir()
            vendor_file.write_text("{}\n")
            with (
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch(
                    "tools.d4rl_runtime.ctypes.util.find_library",
                    return_value="libEGL_nvidia.so.0",
                ),
                mock.patch(
                    "tools.d4rl_runtime.Path.glob",
                    return_value=[],
                ),
            ):
                os.environ.pop(EGL_VENDOR_ENV, None)
                selected = configure_nvidia_egl_vendor(repo_root)

                self.assertEqual(selected, vendor_file)
                self.assertEqual(os.environ[EGL_VENDOR_ENV], str(vendor_file))
                os.environ.pop(EGL_VENDOR_ENV, None)


if __name__ == "__main__":
    unittest.main()
