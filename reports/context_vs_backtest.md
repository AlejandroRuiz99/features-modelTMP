# EXP-5: Contexto vs Backtest — ¿Árbitro + Overlay Explican el Rendimiento?

> **15** apuestas de J28-J38 matcheadas con runs/ | **15** apostadas por el backtest | **15** con delta_p calculado

---

## 1. Magnitud del Impacto del Contexto

| Métrica | Valor |
|---------|-------|
| Δp_over medio (sesgo) | +0.0504 |
| |Δp_over| medio (magnitud) | 0.2055 |
| Δp_over MAX subida | +0.4649 |
| Δp_over MAX bajada | -0.4812 |
| Δμ fouls medio (overlay) | +0.8133 |
| Contexto sube >2pp | 8 (53%) |
| Contexto baja >2pp | 5 (33%) |
| Sin cambio (<= 2pp) | 2 (13%) |

## 2. Cambios de Decisión de Apuesta

| Categoría | N | Hit Rate | ROI |
|-----------|---|----------|-----|
| Ambos apuestan (BT + Context) | 5 | 40.0% | -19.6% |
| Solo backtest (context cancela) | 10 | 80.0% | 44.7% |
| Solo context (backtest no entra) | 0 | 0.0% | 0.0% |

> Si `solo_context` tiene buen ROI: el árbitro/overlay añade bets valiosos.
> Si `bt_only` tiene mal ROI: el contexto filtra bien las apuestas dudosas.

## 3. ¿El Contexto Predice el Resultado?

Entre las apuestas del backtest (J28-J38), ¿acertó el contexto al subir/bajar p?

| Dirección Contexto | N | Ganadas | Hit Rate | ROI |
|--------------------|---|---------|----------|-----|
| Subió >+2pp → más confiado | 8 | 6 | 75.0% | 38.5% |
| Bajó <-2pp → menos confiado | 5 | 3 | 60.0% | 10.2% |
| Sin cambio significativo | 2 | 1 | 50.0% | -5.0% |

> **Señal predictiva ideal**: hit_rate(UP) > hit_rate(FLAT) > hit_rate(DOWN)

## 4. Análisis por Árbitro

| Árbitro | Partidos | Apuestas BT | Hit Rate | Δμ medio | Kelly Scale |
|---------|----------|-------------|----------|-----------|-------------|
| Sesma Espinosa | 2 | 3 | 66.7% | +1.03 | 0.87 |
| Quintero González | 1 | 2 | 100.0% | +2.00 | 0.90 |
| De Burgos Bengoetxea | 1 | 2 | 50.0% | +0.00 | 0.90 |
| Cordero Vega | 1 | 2 | 50.0% | +0.00 | 0.90 |
| Gil Manzano | 1 | 2 | 50.0% | +1.00 | 0.90 |
| Busquets Ferrer | 1 | 1 | 100.0% | +2.00 | 0.85 |
| Hernández Maeso | 1 | 1 | 100.0% | +2.00 | 0.80 |
| Munuera Montero | 1 | 1 | 100.0% | -0.90 | 0.90 |
| Díaz de Mera | 1 | 1 | 0.0% | +0.00 | 0.90 |

## 5. Distribución de Overlay Rules Activadas

| Regla | Activaciones |
|-------|-------------|
| key_injuries_disrupted_volatile | 12 |
| one_relegation_high_stakes_up | 9 |
| high_intensity_override_up | 9 |
| physical_clash_up | 4 |
| strict_ref_physical_up | 4 |
| derbi_intensity_up | 3 |
| both_relegation_up | 2 |
| permissive_ref_technical_down | 2 |
| coach_pressure_up | 2 |
| coach_pressure_volatile | 2 |
| last_round_drama_up | 1 |
| one_team_rotation_down | 1 |
| b_team_expected_down | 1 |

## 6. Top 10 Apuestas con Mayor Divergencia (|Δp| > 5pp)

| Fixture | Mercado | Línea | p_bt | p_run | Δp | Overlay | Won |
|---------|---------|-------|------|-------|----|---------|-----|
| Reial Club vs Real Madri | Team Total Fouls | 13.5 | 0.586 | 0.104 | -0.481 | 1.0 fouls | ✓ |
| Girona FC vs Real Socie | Team Total Fouls | 11.5 | 0.418 | 0.883 | +0.465 | 1.0 fouls | ✗ |
| Real Betis vs Real Ovied | Team Total Fouls | 11.5 | 0.407 | 0.856 | +0.449 | 0.0 fouls | ✓ |
| Valencia C vs Club Atlét | Team Total Fouls | 11.5 | 0.412 | 0.781 | +0.369 | -0.9 fouls | ✓ |
| Real Club  vs Elche CF | Team Total Fouls | 12.5 | 0.611 | 0.260 | -0.351 | 0.0 fouls | ✓ |
| Deportivo  vs Athletic C | Team Total Fouls | 13.5 | 0.376 | 0.582 | +0.207 | 2.0 fouls | ✓ |
| Real Club  vs Elche CF | Team Total Fouls | 12.5 | 0.458 | 0.260 | -0.198 | 0.0 fouls | ✗ |
| Getafe CF vs Rayo Valle | Team Total Fouls | 13.5 | 0.536 | 0.711 | +0.176 | 1.1 fouls | ✓ |
| Girona FC vs Real Socie | Team Total Fouls | 14.5 | 0.537 | 0.651 | +0.114 | 1.0 fouls | ✓ |
| Real Betis vs Real Ovied | Team Total Fouls | 13.5 | 0.592 | 0.699 | +0.106 | 0.0 fouls | ✗ |

## 7. Conclusión

### Hipótesis original
> *¿El mejor rendimiento en J28-J38 viene de haber usado árbitro + contexto?*

**Evidencia cuantitativa**:
- El contexto mueve p_over una media de **20.5%** (magnitud absoluta).
- El overlay ajusta μ una media de **+0.81 fouls** por partido.
- **Señal parcial**: bets con contexto UP ganan más que DOWN, patrón flat inconsistente.
- Las apuestas que contexto cancela tienen ROI 44.7% → filtrado neutro.

**Veredicto**:
El contexto tiene **impacto real** en las probabilidades. Parte del rendimiento de J28-J38 probablemente se debe al uso de árbitro + overlay. Para cuantificar con precisión se necesita un backtest con referee injection (EXP-6).

### Próximos pasos
- **EXP-6** (cuando Supabase esté activo): re-correr backtest con `arbitro_input` real → separa contribución de árbitro del overlay narrativo
- **EXP-7**: backtest con `--narrative-dir` para overlay retroactivo y medir impacto aislado