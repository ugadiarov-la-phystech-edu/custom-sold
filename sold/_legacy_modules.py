"""Module names from before sold/ became a package, so that old checkpoints can be unpickled.

Checkpoints pickle classes by module path: modeling.* (hyperparameters), datasets.info (replay buffer) and the SAVi
scheduler (utils.schedulers, or lightning-bolts' pl_bolts.optimizers.lr_scheduler). LegacyModuleFinder maps these names
to the same module objects under sold, unless a package of that name is installed (e.g. Hugging Face datasets).
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType


def _dotted_prefixes(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts))]


# Old top-level packages, with their submodules.
PACKAGE_ALIASES = {"datasets": "sold.data", "modeling": "sold.modeling", "utils": "sold.utils"}
# Single old modules, under empty packages (SAVi checkpoints name lightning-bolts' scheduler).
MODULE_ALIASES = {"pl_bolts.optimizers.lr_scheduler": "sold.utils.schedulers"}
EMPTY_PACKAGES = {prefix for name in MODULE_ALIASES for prefix in _dotted_prefixes(name)}
SOLD_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _target(fullname: str) -> str:
    """The module under sold that the old name stands for."""
    if fullname in MODULE_ALIASES:
        return MODULE_ALIASES[fullname]
    top, _, rest = fullname.partition(".")
    return PACKAGE_ALIASES[top] + (f".{rest}" if rest else "")


class LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # First in sys.meta_path, or e.g. modeling.sold would be found via the aliased parent's __path__ and imported
    # again under its old name, duplicating classes.

    def find_spec(self, fullname: str, path=None, target=None) -> importlib.machinery.ModuleSpec | None:
        top = fullname.partition(".")[0]
        if not (top in PACKAGE_ALIASES or fullname in MODULE_ALIASES or fullname in EMPTY_PACKAGES):
            return None
        if fullname != top:  # Submodules only below parents aliased by this finder.
            parent = sys.modules.get(top)
            if parent is None or (parent.__name__ == top and not isinstance(parent.__loader__, LegacyModuleFinder)):
                return None
        else:  # An installed package wins; sold/ is on sys.path when scripts run from it.
            search_path = [entry for entry in sys.path if os.path.abspath(entry) + os.sep != SOLD_DIR]
            if importlib.machinery.PathFinder.find_spec(fullname, search_path) is not None:
                return None
        return importlib.util.spec_from_loader(fullname, self, is_package=fullname in EMPTY_PACKAGES)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        if spec.name in EMPTY_PACKAGES:
            return None  # The import system creates an empty package.
        return importlib.import_module(_target(spec.name))

    def exec_module(self, module: ModuleType) -> None:
        pass  # The sold module is imported already, or the package is empty.


def install() -> None:
    if not any(isinstance(finder, LegacyModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, LegacyModuleFinder())
