"""Build and install the wheel outside the checkout, then check policy loading.

Run with `uv run tools/check_wheel.py`. Only dependencies from the current
Python environment are reused; tiermod itself must come from the built wheel.
"""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CHECK = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tiermod import t0
assert Path(t0.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert t0.load_blocklist() == {"zzsynthslur", "zzsynthslur2"}
assert t0.score("zzsynthslur").toxic
t0._BLOCKLIST = None
os.environ["BLOCKLIST_PATH"] = str(Path.cwd() / "missing.json")
try:
    t0.load_blocklist()
except t0.BlocklistError:
    pass
else:
    raise AssertionError("Missing policy was silently accepted")
print("Installed wheel: packaged blocklist and configuration failure verified")
"""


def main():
    with TemporaryDirectory(prefix="tiermod-wheel-") as directory:
        work = Path(directory)
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(work)],
                       cwd=ROOT, check=True)
        wheel = next(work.glob("*.whl"))
        target = work / "installed"
        subprocess.run(["uv", "pip", "install", "--no-deps", "--target", str(target),
                        str(wheel)], check=True)
        env = os.environ.copy()
        env.pop("BLOCKLIST_PATH", None)
        subprocess.run([sys.executable, "-I", "-c", CHECK, str(target)],
                       cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()
