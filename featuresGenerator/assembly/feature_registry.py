"""FeatureRegistry — Carga y aplica la configuracion de features.yaml.

Responsabilidades:
  - Cargar features.yaml y resolver el perfil activo.
  - Validar dependencias: si un grupo derived esta activo, sus depends_on
    tambien deben estarlo. Emite advertencias pero no falla.
  - Filtrar el flat dict de salida omitiendo features de grupos desactivados.
  - Manejar graceful_degradation: si un grupo tiene este flag y sus features
    no estan en el dict, se omiten silenciosamente.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "features.yaml"


class FeatureRegistry:
    """Gestiona que grupos y features estan activos segun el perfil configurado."""

    def __init__(
        self,
        yaml_path: str | Path | None = None,
        profile: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            path = Path(yaml_path) if yaml_path else _DEFAULT_YAML
            with open(path, encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f)

        self._active_profile = profile or self._cfg.get("active_profile", "prediction")
        self._groups_cfg: dict[str, dict] = self._cfg.get("feature_groups", {})
        self._profiles_cfg: dict[str, dict] = self._cfg.get("profiles", {})

        self._enabled_groups = self._resolve_enabled_groups()
        self._enabled_features = self._resolve_enabled_features()
        self._validate_dependencies()

    # ------------------------------------------------------------------
    # Construccion interna
    # ------------------------------------------------------------------

    def _resolve_enabled_groups(self) -> set[str]:
        profile_data = self._profiles_cfg.get(self._active_profile, {})
        groups_in_profile = profile_data.get("groups", {})
        enabled = set()
        for group_name in self._groups_cfg:
            override = groups_in_profile.get(group_name, {})
            if override.get("enabled", True):
                enabled.add(group_name)
        return enabled

    def _resolve_enabled_features(self) -> set[str]:
        features: set[str] = set()
        for group_name, group_def in self._groups_cfg.items():
            if group_name in self._enabled_groups:
                features.update(group_def.get("features", []))
        return features

    def _validate_dependencies(self) -> None:
        """Emite advertencias si un grupo derived tiene dependencias desactivadas."""
        for group_name, group_def in self._groups_cfg.items():
            if group_name not in self._enabled_groups:
                continue
            for dep in group_def.get("depends_on", []):
                if dep not in self._enabled_groups:
                    warnings.warn(
                        f"[FeatureRegistry] ADVERTENCIA: El grupo '{group_name}' (type: "
                        f"{group_def.get('type', '?')}) esta activo pero su dependencia "
                        f"'{dep}' NO lo esta. Esto puede causar features incompletas o errores.",
                        stacklevel=2,
                    )
                    logger.warning(
                        "Grupo '%s' activo con dependencia '%s' desactivada.", group_name, dep
                    )

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    @property
    def active_profile(self) -> str:
        return self._active_profile

    def enabled_groups(self) -> list[str]:
        """Lista de grupos activos en el perfil actual."""
        return sorted(self._enabled_groups)

    def enabled_features(self) -> set[str]:
        """Conjunto de features activas (union de todos los grupos activos)."""
        return set(self._enabled_features)

    def should_compute_group(self, group: str) -> bool:
        """True si el grupo esta activo en el perfil actual."""
        return group in self._enabled_groups

    def is_graceful(self, group: str) -> bool:
        """True si el grupo soporta degradacion elegante (no falla si los datos no existen)."""
        return bool(self._groups_cfg.get(group, {}).get("graceful_degradation", False))

    def filter(self, flat_dict: dict[str, Any]) -> dict[str, Any]:
        """Filtra el flat dict devolviendo solo las features activas.

        Features de grupos con graceful_degradation se omiten si no estan
        presentes en flat_dict (en lugar de incluir None).
        """
        result: dict[str, Any] = {}
        missing: list[str] = []

        for group_name, group_def in self._groups_cfg.items():
            if group_name not in self._enabled_groups:
                continue
            is_graceful = group_def.get("graceful_degradation", False)
            for feature in group_def.get("features", []):
                if feature in flat_dict:
                    result[feature] = flat_dict[feature]
                elif not is_graceful:
                    missing.append(feature)

        if missing:
            logger.debug("Features ausentes en flat_dict: %s", missing)

        return result

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FeatureRegistry(profile={self._active_profile!r}, "
            f"active_groups={len(self._enabled_groups)}, "
            f"active_features={len(self._enabled_features)})"
        )
