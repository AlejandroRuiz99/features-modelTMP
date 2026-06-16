# Simulación Monte Carlo — foultsPredictor · `backtest_all_e05_referee`

_Generado: 2026-06-11 13:48_

- Apuestas empíricas: **453**
- Bankroll inicial normalizado: **1500u**
- Apuestas por camino: **150**
- Caminos por fracción: **10,000**
- Umbral de ruina: **<40.0% del bankroll inicial**

## 1. Comparativa de Fracciones de Kelly

| Fracción Kelly | ROI mediano | ROI p5 | P(ruina) | MaxDD p95 | Bankroll mediano | Score |
|----------------|-------------|--------|----------|-----------|-----------------|-------|
| 0.10x | 14.3% | -8.7% | 0.0% | 9.9% | 1714.4u | 5/100 |
| 0.20x | 28.3% | -18.2% | 0.0% | 19.3% | 1924.2u | 7/100 | ← **ÓPTIMO**
| 0.25x | 35.0% | -23.1% | 0.0% | 23.7% | 2025.1u | 6/100 |
| 0.33x | 45.2% | -31.0% | 0.5% | 30.4% | 2177.7u | 5/100 |
| 0.50x | 63.0% | -60.1% | 5.2% | 43.5% | 2444.6u | 0/100 |
| 0.75x | 70.7% | -62.4% | 19.7% | 59.7% | 2560.5u | 0/100 |
| 1.00x | 36.6% | -64.3% | 35.8% | 68.0% | 2049.5u | 0/100 |
| 1.50x | -61.1% | -68.1% | 62.0% | 77.6% | 583.3u | 0/100 |
| 2.00x | -63.0% | -71.6% | 79.2% | 81.7% | 554.5u | 0/100 |

## 2. Detalle Fracción Óptima: 0.20x Kelly

| Métrica | Valor |
|---------|-------|
| Fracción de Kelly aplicada | 0.20x |
| ROI mediano | 28.28% |
| ROI media | 32.64% |
| ROI p5 (escenario adverso) | -18.19% |
| ROI p95 (escenario favorable) | 97.93% |
| Bankroll terminal mediano | 1924.2u (+28.3%) |
| Bankroll terminal p5 | 1227.2u |
| Bankroll terminal p95 | 2969.0u |
| P(ruina) | 0.00% |
| Max Drawdown mediano | 19.3% |
| Max Drawdown p95 | 34.9% |
| Sharpe mediano (per-bet) | 0.08 |
| Score compuesto | 7/100 |

## 3. Análisis de Riesgo

```
  Fracción |   P(ruina) |  MaxDD p95 |  Bankroll p5
----------------------------------------------------
      0.10x |       0.0% |      18.9% |     1370.0u
      0.20x |       0.0% |      34.9% |     1227.2u
      0.25x |       0.0% |      41.9% |     1153.3u
      0.33x |       0.5% |      51.9% |     1034.6u
      0.50x |       5.2% |      66.0% |      598.8u
      0.75x |      19.7% |      77.6% |      563.3u
      1.00x |      35.8% |      85.0% |      535.8u
      1.50x |      62.0% |      93.3% |      478.9u
      2.00x |      79.2% |      96.8% |      425.7u
```

## 4. Distribución Bankroll Terminal (percentiles)

| Fracción | p5 | p25 | Mediana | p75 | p95 |
|----------|----|-----|---------|-----|-----|
| 0.10x | 1370u | 1564u | 1714u | 1875u | 2129u |
| 0.20x | 1227u | 1602u | 1924u | 2304u | 2969u |
| 0.25x | 1153u | 1610u | 2025u | 2536u | 3482u |
| 0.33x | 1035u | 1609u | 2178u | 2930u | 4459u |
| 0.50x | 599u | 1535u | 2445u | 3826u | 7262u |
| 0.75x | 563u | 1065u | 2561u | 5120u | 13533u |
| 1.00x | 536u | 584u | 2049u | 5908u | 22488u |
| 1.50x | 479u | 544u | 583u | 4002u | 40599u |
| 2.00x | 426u | 507u | 555u | 594u | 41209u |

## 5. Fracción Óptima por Mercado

> Simulado con la fracción óptima global. Útil para ajustar kelly_scale por tipo de mercado.

| Mercado | n bets | ROI mediano | P(ruina) | MaxDD p95 | Score |
|---------|--------|-------------|----------|-----------|-------|
| Team Total Fouls | 453 | 31.7% | 0.0% | 34.2% | 8/100 |
| Total Fouls | 453 | 20.5% | 0.0% | 35.7% | 5/100 |

## 6. Recomendaciones

**Fracción de Kelly recomendada: 0.20x**

- Fracción conservadora. Adecuada dado el tamaño de muestra. Revisar cuando n > 100 en CODERE real.
- ✅ P(ruina) del 0.0% — riesgo controlado.

**Traducción a staking.yaml**: multiplica `bankroll_share_per_stake_unit` por 0.20 para alinear el sistema de staking con la fracción óptima.

## 7. Advertencias Metodológicas

- La distribución empírica tiene **453 apuestas** — remuestreo con reemplazo puede sobreestimar consistencia del edge.
- La función `full_kelly` asume que el edge estimado en el backtest es igual al edge real futuro. En la práctica, aplica un descuento adicional.
- Las apuestas del mismo partido están correlacionadas. El simulador las trata como independientes → P(ruina) ligeramente subestimada.
- Con muestra < 50 (especialmente CODERE), los resultados son orientativos. Prioriza fracciones bajas (0.10x–0.25x) hasta ampliar la muestra.

