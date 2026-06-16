# Simulación Monte Carlo — foultsPredictor · `backtest_codere_e05`

_Generado: 2026-06-11 13:49_

- Apuestas empíricas: **24**
- Bankroll inicial normalizado: **1500u**
- Apuestas por camino: **150**
- Caminos por fracción: **10,000**
- Umbral de ruina: **<40.0% del bankroll inicial**

## 1. Comparativa de Fracciones de Kelly

| Fracción Kelly | ROI mediano | ROI p5 | P(ruina) | MaxDD p95 | Bankroll mediano | Score |
|----------------|-------------|--------|----------|-----------|-----------------|-------|
| 0.10x | 48.8% | 24.4% | 0.0% | 4.7% | 2232.0u | 22/100 |
| 0.20x | 118.7% | 52.9% | 0.0% | 9.3% | 3280.7u | 46/100 |
| 0.25x | 163.9% | 68.8% | 0.0% | 11.6% | 3959.0u | 58/100 |
| 0.33x | 254.5% | 96.4% | 0.0% | 15.1% | 5317.8u | 63/100 | ← **ÓPTIMO**
| 0.50x | 546.4% | 163.8% | 0.0% | 22.3% | 9696.1u | 47/100 |
| 0.75x | 1369.1% | 281.6% | 0.0% | 32.2% | 22036.2u | 26/100 |
| 1.00x | 2996.7% | 396.8% | 0.4% | 41.3% | 46450.8u | 8/100 |
| 1.50x | 10658.8% | 263.4% | 3.7% | 57.2% | 161381.5u | 0/100 |
| 2.00x | 25602.0% | -63.3% | 10.7% | 69.0% | 385529.6u | 0/100 |

## 2. Detalle Fracción Óptima: 0.33x Kelly

| Métrica | Valor |
|---------|-------|
| Fracción de Kelly aplicada | 0.33x |
| ROI mediano | 254.52% |
| ROI media | 274.90% |
| ROI p5 (escenario adverso) | 96.43% |
| ROI p95 (escenario favorable) | 524.40% |
| Bankroll terminal mediano | 5317.8u (+254.5%) |
| Bankroll terminal p5 | 2946.5u |
| Bankroll terminal p95 | 9366.0u |
| P(ruina) | 0.00% |
| Max Drawdown mediano | 15.1% |
| Max Drawdown p95 | 24.8% |
| Sharpe mediano (per-bet) | 0.27 |
| Score compuesto | 63/100 |

## 3. Análisis de Riesgo

```
  Fracción |   P(ruina) |  MaxDD p95 |  Bankroll p5
----------------------------------------------------
      0.10x |       0.0% |       8.1% |     1866.3u
      0.20x |       0.0% |      15.6% |     2294.1u
      0.25x |       0.0% |      19.3% |     2532.0u
      0.33x |       0.0% |      24.8% |     2946.5u
      0.50x |       0.0% |      35.6% |     3956.9u
      0.75x |       0.0% |      49.5% |     5724.6u
      1.00x |       0.4% |      60.9% |     7452.2u
      1.50x |       3.7% |      76.9% |     5451.4u
      2.00x |      10.7% |      86.9% |      550.9u
```

## 4. Distribución Bankroll Terminal (percentiles)

| Fracción | p5 | p25 | Mediana | p75 | p95 |
|----------|----|-----|---------|-----|-----|
| 0.10x | 1866u | 2077u | 2232u | 2393u | 2648u |
| 0.20x | 2294u | 2840u | 3281u | 3773u | 4621u |
| 0.25x | 2532u | 3307u | 3959u | 4714u | 6077u |
| 0.33x | 2947u | 4191u | 5318u | 6693u | 9366u |
| 0.50x | 3957u | 6757u | 9696u | 13743u | 22858u |
| 0.75x | 5725u | 12800u | 22036u | 37185u | 79956u |
| 1.00x | 7452u | 22440u | 46451u | 93403u | 258966u |
| 1.50x | 5451u | 51718u | 161381u | 468948u | 2204358u |
| 2.00x | 551u | 66752u | 385530u | 1693034u | 13906582u |

## 5. Fracción Óptima por Mercado

> Simulado con la fracción óptima global. Útil para ajustar kelly_scale por tipo de mercado.

| Mercado | n bets | ROI mediano | P(ruina) | MaxDD p95 | Score |
|---------|--------|-------------|----------|-----------|-------|
| Total Fouls | 24 | 12.5% | 0.0% | 10.2% | 5/100 |
| Team Total Fouls | 24 | 12.1% | 0.0% | 15.7% | 5/100 |

## 6. Recomendaciones

**Fracción de Kelly recomendada: 0.33x**

- Fracción moderada. Buen balance entre crecimiento y control del riesgo.
- ✅ P(ruina) del 0.0% — riesgo controlado.

**Traducción a staking.yaml**: multiplica `bankroll_share_per_stake_unit` por 0.33 para alinear el sistema de staking con la fracción óptima.

## 7. Advertencias Metodológicas

- La distribución empírica tiene **24 apuestas** — remuestreo con reemplazo puede sobreestimar consistencia del edge.
- La función `full_kelly` asume que el edge estimado en el backtest es igual al edge real futuro. En la práctica, aplica un descuento adicional.
- Las apuestas del mismo partido están correlacionadas. El simulador las trata como independientes → P(ruina) ligeramente subestimada.
- Con muestra < 50 (especialmente CODERE), los resultados son orientativos. Prioriza fracciones bajas (0.10x–0.25x) hasta ampliar la muestra.

