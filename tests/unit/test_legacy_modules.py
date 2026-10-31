"""sold._legacy_modules: old checkpoints name modules from before sold/ was a package."""

import importlib.machinery
import pickle
from sold import _legacy_modules
from sold._legacy_modules import LegacyModuleFinder
from sold.modeling.sold import prediction
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


def test_finder_leaves_installed_packages_of_the_same_name_alone(monkeypatch):
    installed = importlib.machinery.ModuleSpec("utils", None, origin="/site-packages/utils/__init__.py")
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", lambda name, path=None, target=None: installed)
    assert LegacyModuleFinder().find_spec("utils") is None


def test_finder_does_not_count_sold_itself_as_an_installed_package(monkeypatch):
    # Scripts run from sold/ put it on sys.path, where utils/ looks like a top-level package.
    monkeypatch.setattr(sys, "path", [_legacy_modules.SOLD_DIR.rstrip("/"), *sys.path])
    finder = LegacyModuleFinder()
    assert finder.find_spec("utils").loader is finder
