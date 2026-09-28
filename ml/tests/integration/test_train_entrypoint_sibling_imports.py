"""Regression test for issue #185.

SageMaker's v3 `ModelTrainer` "Basic Script driver" invokes the entry
script as `python3 training/train.py` with the bundle root as `cwd` --
Python then puts the *script's own directory* (`<bundle
root>/training`) on `sys.path[0]`, not the invocation `cwd`, so
`train.py`'s sibling-package imports (`from data.corpus_io import ...`,
`from evaluation.run import ...`) failed with `ModuleNotFoundError` the
first time a real training job ran under the v3 SDK.

This is exercised via a real subprocess invocation matching the driver's
exact invocation shape. Naively using this repo's own dev venv
(`sys.executable`) would silently pass even without the fix: `uv sync`
installs this project in editable mode, which adds a `.pth` file to the
venv's site-packages unconditionally pointing at this checkout's `ml/`
root -- so `data`/`evaluation`/`training` are importable from *any* cwd
in this venv regardless of the real bug. `-S` (skip `site` module
initialization) disables exactly that `.pth`-file processing while
`PYTHONPATH` pointed straight at site-packages keeps real third-party
dependencies (torch, transformers, ...) importable -- faithfully
reproducing "only requirements.txt is installed, our own project is
not" (the real container's condition) without needing a real container
or a from-scratch venv build in every test run.
"""

import os
import shutil
import subprocess
import sys
import sysconfig

from training.submit_job import build_source_bundle


def test_train_py_resolves_sibling_package_imports_when_invoked_from_bundle_root():
    site_packages = sysconfig.get_path("purelib")
    bundle_dir = build_source_bundle()
    try:
        result = subprocess.run(
            [sys.executable, "-S", "training/train.py", "--help"],
            cwd=bundle_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONPATH": site_packages},
            check=False,
        )
        assert "ModuleNotFoundError" not in result.stderr, result.stderr
        assert result.returncode == 0, result.stderr
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)
