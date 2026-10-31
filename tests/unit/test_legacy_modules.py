"""sold._legacy_modules: old checkpoints name modules from before sold/ was a package."""

import importlib.machinery
import pickle
from sold import _legacy_modules
from sold._legacy_modules import LegacyModuleFinder
from sold.modeling.sold import prediction
from sold.utils import schedulers
import sys
import types


def test_old_module_paths_unpickle_to_the_current_classes():
    # A pre-package checkpoint's name for GaussianPredictor (pickle protocol 0).
    assert pickle.loads(b"cmodeling.sold.prediction\nGaussianPredictor\n.") is prediction.GaussianPredictor
    assert sys.modules["modeling.sold.prediction"] is prediction  # One module object, no duplicate classes.


def test_finder_ignores_other_names_and_unaliased_parents(monkeypatch):
    finder = LegacyModuleFinder()
    assert finder.find_spec("numpy") is None
    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))  # e.g. Hugging Face datasets
    assert finder.find_spec("datasets.info") is None
    monkeypatch.delitem(sys.modules, "pl_bolts", raising=False)
    assert finder.find_spec("pl_bolts.optimizers") is None  # Parent not imported.


def test_finder_leaves_installed_packages_of_the_same_name_alone(monkeypatch):
    installed = importlib.machinery.ModuleSpec("utils", None, origin="/site-packages/utils/__init__.py")
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", lambda name, path=None, target=None: installed)
    assert LegacyModuleFinder().find_spec("utils") is None


def test_finder_does_not_count_sold_itself_as_an_installed_package(monkeypatch):
    # Scripts run from sold/ put it on sys.path, where utils/ looks like a top-level package.
    monkeypatch.setattr(sys, "path", [_legacy_modules.SOLD_DIR.rstrip("/"), *sys.path])
    finder = LegacyModuleFinder()
    assert finder.find_spec("utils").loader is finder


def test_lightning_bolts_scheduler_without_lightning_bolts(monkeypatch):
    find_spec = importlib.machinery.PathFinder.find_spec
    monkeypatch.setattr(
        importlib.machinery.PathFinder,
        "find_spec",
        lambda name, path=None, target=None: None if name == "pl_bolts" else find_spec(name, path, target),
    )
    for name in [name for name in sys.modules if name.partition(".")[0] == "pl_bolts"]:
        monkeypatch.delitem(sys.modules, name)
    # SAVi checkpoints' name for their scheduler.
    scheduler = pickle.loads(b"cpl_bolts.optimizers.lr_scheduler\nLinearWarmupCosineAnnealingLR\n.")
    assert scheduler is schedulers.LinearWarmupCosineAnnealingLR
    assert LegacyModuleFinder().find_spec("pl_bolts.models") is None  # Only the scheduler's modules are aliased.
