# Certificación — foultsPredictor · `backtest_codere_e05`

_Generado: 2026-06-11 12:36_  |  _Fuente: reports\backtest_codere_e05.csv_

## 1. Resumen Ejecutivo

**🟡 BAJO VIGILANCIA — ROI positivo pero sin significancia aún**

| Indicador | Valor | Umbral | Estado |
|-----------|-------|--------|--------|
| ROI       | 23.88% | >5%  | ✅ |
| p-valor   | 0.1011  | <0.10 | ❌ |
| IC 95% inferior | -13.33% | >-3% | ❌ |
| Muestra ≥ 80 | 24 | ≥80 | ⚠️ |

> ⚠️ **Muestra pequeña** (24 apuestas). Para p<5% con el ROI observado se necesitan ≈ **38 apuestas**.

## 2. Métricas Globales

| Métrica | Valor |
|---------|-------|
| Apuestas (n) | 24 |
| Ganadas / Perdidas | 16 / 8 |
| Hit rate observada | 66.7% |
| Tasa implícita mercado (vig-free) | 53.7% |
| Edge vs mercado (hit rate delta) | 12.97% |
| Odds promedio | 1.87 |
| PnL total | +5.73 u |
| ROI por apuesta | 23.88% |
| Std PnL (per bet) | 0.90 |
| Sharpe (per-bet, sin anualizacion) | 0.27 |
| Max Drawdown | 2.15 u |
| Calmar ratio (ROI/MaxDD) | 0.11 |
| Profit Factor | 1.72 |
| Brier Score | 0.2250 |

## 3. Significancia Estadística

### 3.1 Test de Poisson-Binomial

H₀: para cada apuesta, la probabilidad de victoria = 1/odds (el modelo no bate al mercado).  
H₁: hit_rate real > tasa vig-free (el modelo tiene edge).

- Victorias esperadas bajo H₀: **12.9**
- Victorias observadas: **16**
- Z-estadístico: **1.275**
- p-valor (cola superior): **0.1011**
- Significancia: **—   no sig**

### 3.2 Bootstrap 95% CI sobre ROI (10 000 remuestreos)

- IC: **[-13.33%, 56.75%]**
- ❌ El límite inferior es negativo → edge no demostrado aún.

## 4. Calibración del Modelo (Reliability Diagram)

> **Nota**: solo vemos apuestas con edge ≥ 5% → las probabilidades bajas (<0.55) no aparecen. Sesgo de selección esperado.

| Bucket p_modelo | n | p_esperada | p_observada | Δ | Calibrado? |
|-----------------|---|-----------|------------|---|------------|
| [50-55%) | 5 | 52.5% | 40.0% | -0.125 | ❌ |
| [55-60%) | 14 | 57.5% | 71.4% | +0.139 | ❌ |
| [60-65%) | 5 | 62.5% | 80.0% | +0.175 | ❌ |

**Brier Score: 0.2250**  (referencia aleatoria ≈ 0.25 · referencia perfecta = 0.00)

⚠️  Calibración aceptable.

## 5. Análisis por Segmento

### 5.1 Por Fuente

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| CODERE | 24 | 66.7% | 23.88% | +5.73u | 0.27 | —   no sig |

### 5.2 Por Mercado

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Team Total Fouls | 16 | 62.5% | 16.44% | +2.63u | 0.18 | —   no sig |
| Total Fouls | 8 | 75.0% | 38.75% | +3.10u | 0.45 | —   no sig |

### 5.3 Por Lado (over/under)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| over | 13 | 61.5% | 14.23% | +1.85u | 0.15 | —   no sig |
| under | 11 | 72.7% | 35.27% | +3.88u | 0.41 | —   no sig |

### 5.4 Por Posición (local/visitante/total)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| visitante | 9 | 55.6% | 4.11% | +0.37u | 0.04 | —   no sig |
| total | 8 | 75.0% | 38.75% | +3.10u | 0.45 | —   no sig |
| local | 7 | 71.4% | 32.29% | +2.26u | 0.36 | —   no sig |

### 5.5 Por Bucket de Edge

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| [5-7%) | 7 | 42.9% | -24.29% | -1.70u | -0.26 | —   no sig |
| [7-10%) | 11 | 81.8% | 54.73% | +6.02u | 0.71 | ★★  p<5% |
| [10-13%) | 6 | 66.7% | 23.50% | +1.41u | 0.25 | —   no sig |

### 5.6 Top Equipos por ROI (min 5 apuestas)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Girona FC | 5 | 80.0% | 50.60% | +2.53u | 0.60 | —   no sig |
| Real Madrid CF | 5 | 60.0% | 7.40% | +0.37u | 0.08 | —   no sig |
| Real Sociedad de Fútbol | 5 | 40.0% | -23.00% | -1.15u | -0.22 | —   no sig |

### 5.7 Peores Equipos por ROI

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Real Sociedad de Fútbol | 5 | 40.0% | -23.00% | -1.15u | -0.22 | —   no sig |
| Real Madrid CF | 5 | 60.0% | 7.40% | +0.37u | 0.08 | —   no sig |
| Girona FC | 5 | 80.0% | 50.60% | +2.53u | 0.60 | —   no sig |

### 5.8 Por Jornada (curva PnL acumulado)

| Jornada | n | Hit% | PnL jornada | PnL acumulado |
|---------|---|------|-------------|---------------|
| J28 | 8 | 87.5% | +4.77u | +4.77u 📈 |
| J29 | 4 | 50.0% | -0.35u | +4.42u 📈 |
| J32 | 2 | 50.0% | -0.10u | +4.32u 📈 |
| J34 | 8 | 62.5% | +1.41u | +5.73u 📈 |
| J36 | 2 | 50.0% | +0.00u | +5.73u 📈 |

## 6. Distribución de Edges

`[5-7%)`    7 bets (29.2%)  ██████████

`[7-10%)`   11 bets (45.8%)  ████████████████

`[10-13%)`    6 bets (25.0%)  ████████

## 7. Riesgos y Advertencias

- ⚠️ **Muestra muy pequeña** (24 bets). Resultados indicativos, no concluyentes.
- ℹ️ **Selección bias en calibración**: solo hay apuestas con edge ≥ 5%, por lo que el reliability diagram solo cubre la cola superior de p_model.
- ℹ️ **Correlación intra-partido**: hasta 3 apuestas por fixture (total + local + visitante). Los tests estadísticos asumen independencia → ligeramente optimistas.

## 8. Conclusiones y Próximos Pasos

- ✅ ROI 23.88% → edge claro sobre el mercado.
- ⚠️  Sin significancia estadística (p=0.1011) — necesita ≈ 38 apuestas para confirmar.
- ❌ IC bootstrapped negativo — edge no probado.
- 🏆 Mejor segmento: **Total Fouls** (ROI 38.75%, n=8).

**Siguientes pasos recomendados:**

1. Ejecutar `staking_sim.py` para encontrar la fracción de Kelly óptima.
2. Ampliar muestra: re-ejecutar `backtest_odds.py --source ALL` con Supabase activo.
3. Integrar overlay narrativo en el backtest para comparar pre vs post overlay.
4. Si ROI CODERE se mantiene >10% con n>80 → verde para producción.

