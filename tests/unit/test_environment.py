"""Environment checks for BUGS_AND_PITFALLS.md items 1 and 3a-3c (they pass in a correctly set-up mof-text env)."""

import importlib
import importlib.metadata as metadata

import pytest
from packaging.version import Version

from tests.conftest import CHECKPOINTS_DIR, is_lfs_pointer


def test_01_checkpoints_are_real_files_not_lfs_pointers():
    checkpoints = sorted(CHECKPOINTS_DIR.rglob("*.ckpt"))
    assert checkpoints, "no checkpoints found"
    pointers = [str(path.relative_to(CHECKPOINTS_DIR)) for path in checkpoints if is_lfs_pointer(path)]
    assert not pointers, f"Git LFS pointers instead of weights (run 'git lfs pull'): {pointers}"


def test_03_pinned_versions():
    # 3a: OnlineProgressBar crashes with lightning >= 2.5.3; 3b: rich >= 14.1 crashes lightning 2.5.2.
    assert Version(metadata.version("lightning")) < Version("2.5.3")
    assert Version(metadata.version("rich")) < Version("14.1")
    # 3c: lightning-bolts imports pkg_resources, removed in setuptools 81.
    assert Version(metadata.version("setuptools")) < Version("81")
    import pkg_resources  # noqa: F401
    import pl_bolts.optimizers.lr_scheduler  # noqa: F401


@pytest.mark.parametrize("module", ["torch", "torchvision", "lightning", "pl_bolts", "hydra", "omegaconf", "rich",
                                    "termcolor", "matplotlib", "tensorboard", "av", "gym", "mujoco_py",
                                    "multi_object_fetch"])
def test_required_packages_import(module):
    importlib.import_module(module)
