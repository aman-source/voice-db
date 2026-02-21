import os
import shutil
from pathlib import Path

def apply_windows_symlink_fix():
    """
    Replace Path.symlink_to with a safe copy-based fallback on Windows.
    """
    if os.name != "nt":
        return

    def _safe_symlink(self, target, target_is_directory=False):
        target = Path(target)

        if self.exists():
            return

        self.parent.mkdir(parents=True, exist_ok=True)

        if target.is_dir():
            shutil.copytree(target, self)
        else:
            shutil.copy2(target, self)

    Path.symlink_to = _safe_symlink
