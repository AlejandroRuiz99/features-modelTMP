# Certificación — foultsPredictor · `backtest_all_e05`

_Generado: 2026-06-11 12:36_  |  _Fuente: reports\backtest_all_e05.csv_

## 1. Resumen Ejecutivo

**❌ NO-GO — ROI negativo**

| Indicador | Valor | Umbral | Estado |
|-----------|-------|--------|--------|
| ROI       | -2.19% | >5%  | ❌ |
| p-valor   | 0.6945  | <0.10 | ❌ |
| IC 95% inferior | -10.59% | >-3% | ❌ |
| Muestra ≥ 80 | 451 | ≥80 | ✅ |

## 2. Métricas Globales

| Métrica | Valor |
|---------|-------|
| Apuestas (n) | 451 |
| Ganadas / Perdidas | 242 / 209 |
| Hit rate observada | 53.7% |
| Tasa implícita mercado (vig-free) | 54.8% |
| Edge vs mercado (hit rate delta) | -1.19% |
| Odds promedio | 1.83 |
| PnL total | -9.87 u |
| ROI por apuesta | -2.19% |
| Std PnL (per bet) | 0.91 |
| Sharpe (per-bet, sin anualizacion) | -0.02 |
| Max Drawdown | 35.62 u |
| Calmar ratio (ROI/MaxDD) | 0.00 |
| Profit Factor | 0.95 |
| Brier Score | 0.2505 |

## 3. Significancia Estadística

### 3.1 Test de Poisson-Binomial

H₀: para cada apuesta, la probabilidad de victoria = 1/odds (el modelo no bate al mercado).  
H₁: hit_rate real > tasa vig-free (el modelo tiene edge).

- Victorias esperadas bajo H₀: **247.4**
- Victorias observadas: **242**
- Z-estadístico: **-0.509**
- p-valor (cola superior): **0.6945**
- Significancia: **—   no sig**

### 3.2 Bootstrap 95% CI sobre ROI (10 000 remuestreos)

- IC: **[-10.59%, 6.16%]**
- ❌ El límite inferior es negativo → edge no demostrado aún.

## 4. Calibración del Modelo (Reliability Diagram)

> **Nota**: solo vemos apuestas con edge ≥ 5% → las probabilidades bajas (<0.55) no aparecen. Sesgo de selección esperado.

| Bucket p_modelo | n | p_esperada | p_observada | Δ | Calibrado? |
|-----------------|---|-----------|------------|---|------------|
| [50-55%) | 62 | 52.5% | 51.6% | -0.009 | ✅ |
| [55-60%) | 217 | 57.5% | 50.2% | -0.073 | ⚠️  |
| [60-65%) | 139 | 62.5% | 60.4% | -0.021 | ✅ |
| [65-70%) | 28 | 67.5% | 53.6% | -0.139 | ❌ |
| [70%+) | 5 | 83.7% | 40.0% | -0.437 | ❌ |

**Brier Score: 0.2505**  (referencia aleatoria ≈ 0.25 · referencia perfecta = 0.00)

❌ Modelo sobreconfiado o mal calibrado.

## 5. Análisis por Segmento

### 5.1 Por Fuente

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| CONSENSUS_CODERE_STYLE | 426 | 53.1% | -3.43% | -14.60u | -0.04 | —   no sig |
| CODERE | 25 | 64.0% | 18.92% | +4.73u | 0.21 | —   no sig |

### 5.2 Por Mercado

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Team Total Fouls | 292 | 55.1% | 0.58% | +1.68u | 0.01 | —   no sig |
| Total Fouls | 159 | 50.9% | -7.26% | -11.55u | -0.08 | —   no sig |

### 5.3 Por Lado (over/under)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| under | 247 | 53.4% | -4.68% | -11.56u | -0.05 | —   no sig |
| over | 204 | 53.9% | 0.83% | +1.69u | 0.01 | —   no sig |

### 5.4 Por Posición (local/visitante/total)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| total | 159 | 50.9% | -7.26% | -11.55u | -0.08 | —   no sig |
| visitante | 156 | 54.5% | -0.69% | -1.08u | -0.01 | —   no sig |
| local | 136 | 55.9% | 2.03% | +2.76u | 0.02 | —   no sig |

### 5.5 Por Bucket de Edge

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| [5-7%) | 163 | 48.5% | -11.53% | -18.79u | -0.13 | —   no sig |
| [7-10%) | 138 | 57.2% | 3.86% | +5.33u | 0.04 | —   no sig |
| [10-13%) | 94 | 57.4% | 4.32% | +4.06u | 0.05 | —   no sig |
| [13-16%) | 39 | 53.8% | 0.13% | +0.05u | 0.00 | —   no sig |
| [16%+) | 17 | 52.9% | -3.06% | -0.52u | -0.03 | —   no sig |

### 5.6 Top Equipos por ROI (min 5 apuestas)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Real Betis Balompié | 53 | 67.9% | 23.72% | +12.57u | 0.28 | ★★  p<5% |
| Athletic Club Bilbao | 53 | 64.2% | 16.04% | +8.50u | 0.18 | ★   p<10% |
| Getafe CF | 50 | 58.0% | 8.36% | +4.18u | 0.09 | —   no sig |
| Real Madrid CF | 68 | 58.8% | 8.35% | +5.68u | 0.09 | —   no sig |
| Valencia CF | 44 | 59.1% | 7.86% | +3.46u | 0.09 | —   no sig |
| Levante UD | 40 | 57.5% | 6.47% | +2.59u | 0.07 | —   no sig |
| Reial Club Deportiu Espanyol | 40 | 57.5% | 5.40% | +2.16u | 0.06 | —   no sig |
| Rayo Vallecano | 42 | 57.1% | 3.60% | +1.51u | 0.04 | —   no sig |
| Real Club Deportivo Mallorca | 41 | 56.1% | 1.02% | +0.42u | 0.01 | —   no sig |
| Deportivo Alavés | 38 | 55.3% | -0.84% | -0.32u | -0.01 | —   no sig |
| Villarreal CF | 37 | 54.1% | -3.38% | -1.25u | -0.04 | —   no sig |
| Sevilla FC | 43 | 53.5% | -3.65% | -1.57u | -0.04 | —   no sig |

### 5.7 Peores Equipos por ROI

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Real Sociedad de Fútbol | 51 | 37.3% | -32.82% | -16.74u | -0.37 | —   no sig |
| FC Barcelona | 48 | 39.6% | -27.94% | -13.41u | -0.31 | —   no sig |
| Real Oviedo | 49 | 44.9% | -19.55% | -9.58u | -0.22 | —   no sig |
| Elche CF | 51 | 47.1% | -12.90% | -6.58u | -0.14 | —   no sig |
| Real Club Celta de Vigo | 37 | 48.6% | -11.16% | -4.13u | -0.12 | —   no sig |
| Girona FC | 37 | 48.6% | -9.78% | -3.62u | -0.10 | —   no sig |
| CA Osasuna | 52 | 51.9% | -4.92% | -2.56u | -0.05 | —   no sig |
| Club Atlético de Madrid | 28 | 53.6% | -3.75% | -1.05u | -0.04 | —   no sig |

### 5.8 Por Jornada (curva PnL acumulado)

| Jornada | n | Hit% | PnL jornada | PnL acumulado |
|---------|---|------|-------------|---------------|
| J1 | 11 | 36.4% | -3.66u | -3.66u 📉 |
| J2 | 16 | 25.0% | -8.67u | -12.33u 📉 |
| J3 | 15 | 46.7% | -2.37u | -14.70u 📉 |
| J4 | 16 | 56.2% | +0.67u | -14.03u 📉 |
| J5 | 13 | 76.9% | +5.78u | -8.25u 📉 |
| J6 | 14 | 64.3% | +2.66u | -5.59u 📉 |
| J7 | 15 | 26.7% | -7.51u | -13.10u 📉 |
| J8 | 14 | 64.3% | +2.69u | -10.41u 📉 |
| J9 | 16 | 50.0% | -1.32u | -11.73u 📉 |
| J10 | 18 | 61.1% | +1.71u | -10.02u 📉 |
| J11 | 17 | 35.3% | -6.36u | -16.38u 📉 |
| J12 | 14 | 71.4% | +4.00u | -12.38u 📉 |
| J13 | 15 | 60.0% | +1.40u | -10.98u 📉 |
| J14 | 13 | 46.2% | -2.07u | -13.05u 📉 |
| J15 | 12 | 16.7% | -8.43u | -21.48u 📉 |
| J16 | 10 | 50.0% | -1.24u | -22.72u 📉 |
| J17 | 14 | 64.3% | +2.31u | -20.41u 📉 |
| J18 | 9 | 44.4% | -1.73u | -22.14u 📉 |
| J19 | 13 | 46.2% | -1.78u | -23.92u 📉 |
| J20 | 10 | 40.0% | -2.75u | -26.67u 📉 |
| J21 | 10 | 60.0% | +0.92u | -25.75u 📉 |
| J22 | 18 | 61.1% | +2.03u | -23.72u 📉 |
| J23 | 9 | 22.2% | -5.47u | -29.19u 📉 |
| J24 | 18 | 44.4% | -3.60u | -32.79u 📉 |
| J25 | 11 | 90.9% | +7.60u | -25.19u 📉 |
| J26 | 9 | 66.7% | +1.98u | -23.21u 📉 |
| J27 | 13 | 46.2% | -2.01u | -25.22u 📉 |
| J28 | 12 | 83.3% | +6.20u | -19.02u 📉 |
| J29 | 16 | 62.5% | +1.94u | -17.08u 📉 |
| J30 | 17 | 70.6% | +4.73u | -12.35u 📉 |
| J31 | 11 | 63.6% | +1.70u | -10.65u 📉 |
| J32 | 5 | 60.0% | +0.39u | -10.26u 📉 |
| J33 | 13 | 46.2% | -2.20u | -12.46u 📉 |
| J34 | 12 | 66.7% | +2.59u | -9.87u 📉 |
| J36 | 2 | 50.0% | +0.00u | -9.87u 📉 |

## 6. Distribución de Edges

`[5-7%)`  163 bets (36.1%)  ████████████

`[7-10%)`  138 bets (30.6%)  ██████████

`[10-13%)`   94 bets (20.8%)  ███████

`[13-16%)`   39 bets (8.6%)  ███

`[16%+)`   17 bets (3.8%)  █

## 7. Riesgos y Advertencias

- ⚠️ **Cuotas sintéticas presentes** (CONSENSUS_CODERE_STYLE). El modelo fue calibrado para ajustarse al mercado → posible sobreajuste en este segmento.
- ℹ️ Dataset mixto: CODERE, CONSENSUS_CODERE_STYLE. Interpretar segmento CODERE como benchmark primario.
- ℹ️ **Selección bias en calibración**: solo hay apuestas con edge ≥ 5%, por lo que el reliability diagram solo cubre la cola superior de p_model.
- ℹ️ **Correlación intra-partido**: hasta 3 apuestas por fixture (total + local + visitante). Los tests estadísticos asumen independencia → ligeramente optimistas.

## 8. Conclusiones y Próximos Pasos

- ❌ ROI -2.19% → sin edge.
- ⚠️  Sin significancia estadística (p=0.6945) — necesita ≈ inf apuestas para confirmar.
- ❌ IC bootstrapped negativo — edge no probado.
- 🏆 Mejor segmento: **Team Total Fouls** (ROI 0.58%, n=292).

**Siguientes pasos recomendados:**

1. Ejecutar `staking_sim.py` para encontrar la fracción de Kelly óptima.
2. Ampliar muestra: re-ejecutar `backtest_odds.py --source ALL` con Supabase activo.
3. Integrar overlay narrativo en el backtest para comparar pre vs post overlay.
4. Si ROI CODERE se mantiene >10% con n>80 → verde para producción.

