# Certificación — foultsPredictor · `all_e05_referee`

_Generado: 2026-06-11 13:36_  |  _Fuente: reports\backtest_all_e05_referee.csv_

## 1. Resumen Ejecutivo

**✅ GO — evidencia estadística de edge positivo**

| Indicador | Valor | Umbral | Estado |
|-----------|-------|--------|--------|
| ROI       | 5.56% | >5%  | ✅ |
| p-valor   | 0.0876  | <0.10 | ✅ |
| IC 95% inferior | -2.64% | >-3% | ✅ |
| Muestra ≥ 80 | 453 | ≥80 | ✅ |

## 2. Métricas Globales

| Métrica | Valor |
|---------|-------|
| Apuestas (n) | 453 |
| Ganadas / Perdidas | 263 / 190 |
| Hit rate observada | 58.1% |
| Tasa implícita mercado (vig-free) | 54.9% |
| Edge vs mercado (hit rate delta) | 3.17% |
| Odds promedio | 1.83 |
| PnL total | +25.18 u |
| ROI por apuesta | 5.56% |
| Std PnL (per bet) | 0.90 |
| Sharpe (per-bet, sin anualizacion) | 0.06 |
| Max Drawdown | 20.01 u |
| Calmar ratio (ROI/MaxDD) | 0.00 |
| Profit Factor | 1.13 |
| Brier Score | 0.2409 |

## 3. Significancia Estadística

### 3.1 Test de Poisson-Binomial

H₀: para cada apuesta, la probabilidad de victoria = 1/odds (el modelo no bate al mercado).  
H₁: hit_rate real > tasa vig-free (el modelo tiene edge).

- Victorias esperadas bajo H₀: **248.7**
- Victorias observadas: **263**
- Z-estadístico: **1.356**
- p-valor (cola superior): **0.0876**
- Significancia: **★   p<10%**

### 3.2 Bootstrap 95% CI sobre ROI (10 000 remuestreos)

- IC: **[-2.64%, 13.78%]**
- ⚠️  El límite inferior roza cero → se necesita más muestra para confirmar.

## 4. Calibración del Modelo (Reliability Diagram)

> **Nota**: solo vemos apuestas con edge ≥ 5% → las probabilidades bajas (<0.55) no aparecen. Sesgo de selección esperado.

| Bucket p_modelo | n | p_esperada | p_observada | Δ | Calibrado? |
|-----------------|---|-----------|------------|---|------------|
| [50-55%) | 61 | 52.5% | 44.3% | -0.082 | ⚠️  |
| [55-60%) | 200 | 57.5% | 56.0% | -0.015 | ✅ |
| [60-65%) | 151 | 62.5% | 65.6% | +0.031 | ✅ |
| [65-70%) | 37 | 67.5% | 59.5% | -0.080 | ⚠️  |
| [70%+) | 4 | 83.7% | 75.0% | -0.087 | ⚠️  |

**Brier Score: 0.2409**  (referencia aleatoria ≈ 0.25 · referencia perfecta = 0.00)

⚠️  Calibración aceptable.

## 5. Análisis por Segmento

### 5.1 Por Fuente

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| CONSENSUS_CODERE_STYLE | 428 | 57.7% | 4.78% | +20.45u | 0.05 | —   no sig |
| CODERE | 25 | 64.0% | 18.92% | +4.73u | 0.21 | —   no sig |

### 5.2 Por Mercado

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Team Total Fouls | 307 | 59.0% | 7.14% | +21.93u | 0.08 | ★   p<10% |
| Total Fouls | 146 | 56.2% | 2.23% | +3.25u | 0.02 | —   no sig |

### 5.3 Por Lado (over/under)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| under | 247 | 60.3% | 7.35% | +18.15u | 0.08 | ★   p<10% |
| over | 206 | 55.3% | 3.41% | +7.03u | 0.04 | —   no sig |

### 5.4 Por Posición (local/visitante/total)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| visitante | 170 | 60.0% | 9.00% | +15.30u | 0.10 | ★   p<10% |
| total | 146 | 56.2% | 2.23% | +3.25u | 0.02 | —   no sig |
| local | 137 | 57.7% | 4.84% | +6.63u | 0.05 | —   no sig |

### 5.5 Por Bucket de Edge

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| [5-7%) | 140 | 51.4% | -6.77% | -9.48u | -0.07 | —   no sig |
| [7-10%) | 147 | 62.6% | 13.29% | +19.53u | 0.15 | ★★  p<5% |
| [10-13%) | 91 | 59.3% | 7.44% | +6.77u | 0.08 | —   no sig |
| [13-16%) | 46 | 56.5% | 4.98% | +2.29u | 0.05 | —   no sig |
| [16%+) | 29 | 65.5% | 20.93% | +6.07u | 0.23 | —   no sig |

### 5.6 Top Equipos por ROI (min 5 apuestas)

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Reial Club Deportiu Espanyol | 41 | 68.3% | 25.49% | +10.45u | 0.29 | ★★  p<5% |
| Rayo Vallecano | 37 | 67.6% | 22.89% | +8.47u | 0.26 | ★   p<10% |
| Athletic Club Bilbao | 52 | 67.3% | 21.19% | +11.02u | 0.25 | ★★  p<5% |
| Real Betis Balompié | 56 | 64.3% | 16.68% | +9.34u | 0.19 | ★   p<10% |
| Club Atlético de Madrid | 34 | 64.7% | 15.62% | +5.31u | 0.18 | —   no sig |
| Real Club Deportivo Mallorca | 44 | 63.6% | 15.18% | +6.68u | 0.17 | —   no sig |
| Real Club Celta de Vigo | 32 | 62.5% | 13.63% | +4.36u | 0.15 | —   no sig |
| Getafe CF | 51 | 60.8% | 12.96% | +6.61u | 0.14 | —   no sig |
| Real Madrid CF | 72 | 61.1% | 12.24% | +8.81u | 0.14 | —   no sig |
| Levante UD | 44 | 59.1% | 9.00% | +3.96u | 0.10 | —   no sig |
| Deportivo Alavés | 35 | 60.0% | 7.37% | +2.58u | 0.08 | —   no sig |
| Sevilla FC | 41 | 58.5% | 5.05% | +2.07u | 0.06 | —   no sig |

### 5.7 Peores Equipos por ROI

| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |
|----------|---|------|-----|-----|--------|-----|
| Villarreal CF | 40 | 37.5% | -32.40% | -12.96u | -0.37 | —   no sig |
| FC Barcelona | 45 | 46.7% | -15.27% | -6.87u | -0.17 | —   no sig |
| Real Sociedad de Fútbol | 50 | 48.0% | -13.58% | -6.79u | -0.15 | —   no sig |
| Elche CF | 54 | 51.9% | -4.87% | -2.63u | -0.05 | —   no sig |
| Girona FC | 37 | 54.1% | -0.65% | -0.24u | -0.01 | —   no sig |
| Valencia CF | 42 | 54.8% | -0.40% | -0.17u | -0.00 | —   no sig |
| CA Osasuna | 53 | 54.7% | -0.36% | -0.19u | -0.00 | —   no sig |
| Real Oviedo | 46 | 56.5% | 1.20% | +0.55u | 0.01 | —   no sig |

### 5.8 Por Jornada (curva PnL acumulado)

| Jornada | n | Hit% | PnL jornada | PnL acumulado |
|---------|---|------|-------------|---------------|
| J1 | 11 | 45.5% | -1.80u | -1.80u 📉 |
| J2 | 12 | 50.0% | -1.02u | -2.82u 📉 |
| J3 | 12 | 66.7% | +2.80u | -0.02u 📉 |
| J4 | 17 | 70.6% | +4.90u | +4.88u 📈 |
| J5 | 17 | 76.5% | +7.34u | +12.22u 📈 |
| J6 | 12 | 75.0% | +4.55u | +16.77u 📈 |
| J7 | 17 | 35.3% | -5.78u | +10.99u 📈 |
| J8 | 17 | 64.7% | +3.13u | +14.12u 📈 |
| J9 | 13 | 69.2% | +3.59u | +17.71u 📈 |
| J10 | 18 | 55.6% | -0.11u | +17.60u 📈 |
| J11 | 17 | 23.5% | -9.84u | +7.76u 📈 |
| J12 | 18 | 66.7% | +3.25u | +11.01u 📈 |
| J13 | 12 | 50.0% | -0.90u | +10.11u 📈 |
| J14 | 14 | 42.9% | -3.07u | +7.04u 📈 |
| J15 | 11 | 36.4% | -3.74u | +3.30u 📈 |
| J16 | 12 | 58.3% | +0.50u | +3.80u 📈 |
| J17 | 15 | 60.0% | +1.28u | +5.08u 📈 |
| J18 | 11 | 45.5% | -2.16u | +2.92u 📈 |
| J19 | 13 | 61.5% | +1.43u | +4.35u 📈 |
| J20 | 9 | 77.8% | +3.78u | +8.13u 📈 |
| J21 | 7 | 42.9% | -1.70u | +6.43u 📈 |
| J22 | 20 | 65.0% | +3.60u | +10.03u 📈 |
| J23 | 11 | 27.3% | -5.68u | +4.35u 📈 |
| J24 | 19 | 47.4% | -2.82u | +1.53u 📈 |
| J25 | 10 | 100.0% | +8.60u | +10.13u 📈 |
| J26 | 10 | 60.0% | +0.83u | +10.96u 📈 |
| J27 | 10 | 50.0% | -1.13u | +9.83u 📈 |
| J28 | 12 | 83.3% | +6.20u | +16.03u 📈 |
| J29 | 16 | 62.5% | +1.94u | +17.97u 📈 |
| J30 | 17 | 70.6% | +4.73u | +22.70u 📈 |
| J31 | 11 | 63.6% | +1.70u | +24.40u 📈 |
| J32 | 5 | 60.0% | +0.39u | +24.79u 📈 |
| J33 | 13 | 46.2% | -2.20u | +22.59u 📈 |
| J34 | 12 | 66.7% | +2.59u | +25.18u 📈 |
| J36 | 2 | 50.0% | +0.00u | +25.18u 📈 |

## 6. Distribución de Edges

`[5-7%)`  140 bets (30.9%)  ██████████

`[7-10%)`  147 bets (32.5%)  ███████████

`[10-13%)`   91 bets (20.1%)  ███████

`[13-16%)`   46 bets (10.2%)  ███

`[16%+)`   29 bets (6.4%)  ██

## 7. Riesgos y Advertencias

- ⚠️ **Cuotas sintéticas presentes** (CONSENSUS_CODERE_STYLE). El modelo fue calibrado para ajustarse al mercado → posible sobreajuste en este segmento.
- ℹ️ Dataset mixto: CODERE, CONSENSUS_CODERE_STYLE. Interpretar segmento CODERE como benchmark primario.
- ℹ️ **Selección bias en calibración**: solo hay apuestas con edge ≥ 5%, por lo que el reliability diagram solo cubre la cola superior de p_model.
- ℹ️ **Correlación intra-partido**: hasta 3 apuestas por fixture (total + local + visitante). Los tests estadísticos asumen independencia → ligeramente optimistas.

## 8. Conclusiones y Próximos Pasos

- ✅ ROI 5.56% → edge positivo.
- ⚠️  Significancia marginal al 10% (p=0.0876).
- ⚠️  IC bootstrapped roza cero — ampliar muestra.
- 🏆 Mejor segmento: **Team Total Fouls** (ROI 7.14%, n=307).

**Siguientes pasos recomendados:**

1. Ejecutar `staking_sim.py` para encontrar la fracción de Kelly óptima.
2. Ampliar muestra: re-ejecutar `backtest_odds.py --source ALL` con Supabase activo.
3. Integrar overlay narrativo en el backtest para comparar pre vs post overlay.
4. Si ROI CODERE se mantiene >10% con n>80 → verde para producción.

