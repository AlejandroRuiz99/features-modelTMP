"""training_data — Pipeline de generacion del dataset de entrenamiento.

Descarga partidos historicos, objectives y calendario desde Supabase,
construye features para cada partido y exporta un Parquet para prediction_models.

Uso:
    python -m training_data
    python -m training_data --output ../prediction_models/data/training.parquet
"""

from .generator import run

__all__ = ["run"]
