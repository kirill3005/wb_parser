from typing import Dict, List, Optional

from .types import FeaturePackage
from ..cache import load_feature_pkg, save_feature_pkg


class FeatureStore:
    def __init__(self) -> None:
        self._pkgs: Dict[str, FeaturePackage] = {}

    def add(self, pkg: FeaturePackage) -> None:
        assert pkg.name not in self._pkgs, f"duplicate package {pkg.name}"
        self._pkgs[pkg.name] = pkg

    def get(self, name: str) -> FeaturePackage:
        return self._pkgs[name]

    def list(self) -> List[str]:
        return list(self._pkgs.keys())

    def catalog(self) -> Dict[str, dict]:
        return {
            k: {"cols": v.cols, "kind": v.kind, "meta": v.meta}
            for k, v in self._pkgs.items()
        }

    def load_cached(self, name: str, key: str) -> Optional[FeaturePackage]:
        return load_feature_pkg(name, key)

    def save_cached(self, name: str, key: str, pkg: FeaturePackage) -> None:
        save_feature_pkg(name, key, pkg)
