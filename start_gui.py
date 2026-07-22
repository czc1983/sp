from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    from ui.main_window import run_app

    preload_project_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    return run_app(preload_project_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")
        raise
