# Simulación Monte Carlo — foultsPredictor · `codere_e05`

_Generado: 2026-06-11 12:37_

- Apuestas empíricas: **24**
- Bankroll inicial normalizado: **100u**
- Apuestas por camino: **200**
- Caminos por fracción: **20,000**
- Umbral de ruina: **<40.0% del bankroll inicial**

## 1. Comparativa de Fracciones de Kelly

| Fracción Kelly | ROI mediano | ROI p5 | P(ruina) | MaxDD p95 | Bankroll mediano | Score |
|----------------|-------------|--------|----------|-----------|-----------------|-------|
| 0.10x | 69.4% | 38.5% | 0.0% | 5.1% | 169.4u | 30/100 |
| 0.20x | 182.4% | 88.8% | 0.0% | 10.0% | 282.4u | 69/100 |
| 0.25x | 262.5% | 119.1% | 0.0% | 12.4% | 362.5u | 70/100 | ← **ÓPTIMO**
| 0.33x | 436.1% | 175.4% | 0.0% | 16.2% | 536.1u | 61/100 |
| 0.50x | 1089.0% | 333.4% | 0.0% | 23.8% | 1189.0u | 44/100 |
| 0.75x | 3427.1% | 674.3% | 0.1% | 34.3% | 3527.1u | 23/100 |
| 1.00x | 9378.5% | 1103.0% | 0.5% | 43.8% | 9478.5u | 5/100 |
| 1.50x | 49312.2% | 819.1% | 3.7% | 60.0% | 49412.2u | 0/100 |
| 2.00x | 154841.1% | -63.1% | 10.5% | 71.7% | 154941.1u | 0/100 |

## 2. Detalle Fracción Óptima: 0.25x Kelly

| Métrica | Valor |
|---------|-------|
| Fracción de Kelly aplicada | 0.25x |
| ROI mediano | 262.48% |
| ROI media | 279.98% |
| ROI p5 (escenario adverso) | 119.06% |
| ROI p95 (escenario favorable) | 495.30% |
| Bankroll terminal mediano | 362.5u (+262.5%) |
| Bankroll terminal p5 | 219.1u |
| Bankroll terminal p95 | 595.3u |
| P(ruina) | 0.00% |
| Max Drawdown mediano | 12.4% |
| Max Drawdown p95 | 20.3% |
| Sharpe mediano (per-bet) | 0.28 |
| Score compuesto | 70/100 |

## 3. Análisis de Riesgo

```
  Fracción |   P(ruina) |  MaxDD p95 |  Bankroll p5
----------------------------------------------------
      0.10x |       0.0% |       8.5% |      138.5u
      0.20x |       0.0% |      16.5% |      188.8u
      0.25x |       0.0% |      20.3% |      219.1u
      0.33x |       0.0% |      26.0% |      275.4u
      0.50x |       0.0% |      37.3% |      433.4u
      0.75x |       0.1% |      51.5% |      774.3u
      1.00x |       0.5% |      63.0% |     1203.0u
      1.50x |       3.7% |      79.3% |      919.1u
      2.00x |      10.5% |      89.0% |       36.9u
```

## 4. Distribución Bankroll Terminal (percentiles)

| Fracción | p5 | p25 | Mediana | p75 | p95 |
|----------|----|-----|---------|-----|-----|
| 0.10x | 139u | 156u | 169u | 184u | 207u |
| 0.20x | 189u | 239u | 282u | 334u | 420u |
| 0.25x | 219u | 295u | 362u | 447u | 595u |
| 0.33x | 275u | 408u | 536u | 706u | 1032u |
| 0.50x | 433u | 786u | 1189u | 1806u | 3209u |
| 0.75x | 774u | 1895u | 3527u | 6605u | 15727u |
| 1.00x | 1203u | 4103u | 9478u | 21893u | 69671u |
| 1.50x | 919u | 13194u | 49412u | 177897u | 1022274u |
| 2.00x | 37u | 20811u | 154941u | 930738u | 10049659u |

## 5. Fracción Óptima por Mercado

> Simulado con la fracción óptima global. Útil para ajustar kelly_scale por tipo de mercado.

| Mercado | n bets | ROI mediano | P(ruina) | MaxDD p95 | Score |
|---------|--------|-------------|----------|-----------|-------|
| Total Fouls | 24 | 9.3% | 0.0% | 7.8% | 4/100 |
| Team Total Fouls | 24 | 9.2% | 0.0% | 12.2% | 4/100 |

## 6. Recomendaciones

**Fracción de Kelly recomendada: 0.25x**

- Fracción conservadora. Adecuada dado el tamaño de muestra. Revisar cuando n > 100 en CODERE real.
- ✅ P(ruina) del 0.0% — riesgo controlado.

**Traducción a staking.yaml**: multiplica `bankroll_share_per_stake_unit` por 0.25 para alinear el sistema de staking con la fracción óptima.

## 7. Advertencias Metodológicas

- La distribución empírica tiene **24 apuestas** — remuestreo con reemplazo puede sobreestimar consistencia del edge.
- La función `full_kelly` asume que el edge estimado en el backtest es igual al edge real futuro. En la práctica, aplica un descuento adicional.
- Las apuestas del mismo partido están correlacionadas. El simulador las trata como independientes → P(ruina) ligeramente subestimada.
- Con muestra < 50 (especialmente CODERE), los resultados son orientativos. Prioriza fracciones bajas (0.10x–0.25x) hasta ampliar la muestra.

