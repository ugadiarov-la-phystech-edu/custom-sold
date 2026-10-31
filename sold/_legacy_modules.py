"""Module names from before sold/ became a package, so that old checkpoints can be unpickled.

Checkpoints pickle objects by module path: the autoencoder and predictor classes of their hyperparameters
(modeling.*), the replay buffer's field types (datasets.info) and the SAVi scheduler (utils.schedulers).
LegacyModuleFinder resolves these names to the module objects under sold, so that old and new names refer to the
same classes, unless a package of the same name is installed (e.g. Hugging Face datasets).
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType

LEGACY_NAMES = {"datasets": "sold.data", "modeling": "sold.modeling", "utils": "sold.utils"}
SOLD_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # First in sys.meta_path, or e.g. modeling.sold would be found via the aliased parent's __path__ and imported
    # again under its old name, duplicating classes.

    def find_spec(self, fullname: str, path=None, target=None) -> importlib.machinery.ModuleSpec | None:
        top = fullname.partition(".")[0]
        if top not in LEGACY_NAMES:
            return None
        if fullname != top:  # A submodule: only below a parent aliased by this finder.
            if getattr(sys.modules.get(top), "__name__", None) != LEGACY_NAMES[top]:
                return None
        else:  # An installed package wins; sold/ is on sys.path when scripts run from it.
            search_path = [entry for entry in sys.path if os.path.abspath(entry) + os.sep != SOLD_DIR]
            if importlib.machinery.PathFinder.find_spec(fullname, search_path) is not None:
                return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        top, _, rest = spec.name.partition(".")
        return importlib.import_module(LEGACY_NAMES[top] + (f".{rest}" if rest else ""))

    def exec_module(self, module: ModuleType) -> None:
        pass  # The module under sold is imported (and executed) already.


def install() -> None:
    if not any(isinstance(finder, LegacyModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, LegacyModuleFinder())
