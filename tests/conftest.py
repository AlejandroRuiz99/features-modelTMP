"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project roots to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_ROOT = PROJECT_ROOT / "features_generator"
PRED_ROOT = PROJECT_ROOT / "prediction_models"

for p in [str(PROJECT_ROOT), str(FEATURES_ROOT), str(PRED_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Add predecir-jornada skill scripts to path for unit tests
_SKILL_SCRIPTS = Path.home() / ".claude" / "skills" / "predecir-jornada" / "scripts"
if _SKILL_SCRIPTS.exists() and str(_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS))
