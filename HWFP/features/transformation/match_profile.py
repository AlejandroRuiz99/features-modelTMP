"""Perfil del partido: compatibilidad de estilos, volumen de eventos y narrative."""

from __future__ import annotations

from HWFP.features.core.config import (
    FISICO_MUY_ALTO,
    FISICO_ALTO,
    FISICO_NORMAL,
    OFENSIVO_MUY_ABIERTO,
    OFENSIVO_ABIERTO,
    OFENSIVO_EQUILIBRADO,
)
from HWFP.features.core.helpers import safe

_FISICO_STYLES = {"FÍSICO-DEFENSIVO", "INTENSO", "DIRECTO-FÍSICO"}
_TECH_STYLES   = {"TÉCNICO-OFENSIVO", "POSESIÓN"}


def calcular_compatibilidad_estilos(
    xstyles: dict, equipo_local: str, equipo_visitante: str,
) -> dict:
    """Clasifica el tipo de partido esperado segun el cruce de perfiles de estilo."""
    sl = xstyles.get(equipo_local,     {})
    sv = xstyles.get(equipo_visitante, {})

    fouls_l   = safe(sl.get("fouls"),    12.0)
    fouls_v   = safe(sv.get("fouls"),    12.0)
    shots_l   = safe(sl.get("tiros"),    11.0)
    shots_v   = safe(sv.get("tiros"),    11.0)
    poss_l    = sl.get("posesion") or 50.0
    poss_v    = sv.get("posesion") or 50.0
    estilo_l  = sl.get("estilo", "EQUILIBRADO")
    estilo_v  = sv.get("estilo", "EQUILIBRADO")

    fisico_total    = fouls_l + fouls_v
    ritmo_ofensivo  = shots_l + shots_v
    dif_posesion    = abs(poss_l - poss_v)

    _FISICO_LABEL  = [(FISICO_MUY_ALTO, "muy alto"), (FISICO_ALTO, "alto"), (FISICO_NORMAL, "normal"), (0, "bajo")]
    _OFENS_LABEL   = [(OFENSIVO_MUY_ABIERTO, "muy abierto"), (OFENSIVO_ABIERTO, "abierto"), (OFENSIVO_EQUILIBRADO, "equilibrado"), (0, "cerrado")]
    fisico_label   = next(l for t, l in _FISICO_LABEL  if fisico_total   >= t)
    ofensivo_label = next(l for t, l in _OFENS_LABEL   if ritmo_ofensivo >= t)
    control_juego  = (
        "dominio claro de un equipo" if dif_posesion > 12
        else "ligero control de un equipo" if dif_posesion > 6
        else "equilibrio en posesion"
    )

    physical  = estilo_l in _FISICO_STYLES or estilo_v in _FISICO_STYLES
    technical = estilo_l in _TECH_STYLES   or estilo_v in _TECH_STYLES
    es_fis_l  = estilo_l in _FISICO_STYLES
    es_fis_v  = estilo_v in _FISICO_STYLES
    es_tec_l  = estilo_l in _TECH_STYLES
    es_tec_v  = estilo_v in _TECH_STYLES

    if physical and technical:                                               tipo, desc = "PARTIDO MIXTO",              "Choque de estilos opuestos, partido imprevisible con transiciones"
    elif physical:                                                           tipo, desc = "PARTIDO FÍSICO",             "Alta intensidad y contacto, bajo ritmo tecnico"
    elif technical:                                                          tipo, desc = "PARTIDO TÉCNICO",            "Predominio del juego elaborado y la posesion"
    elif fisico_total >= FISICO_ALTO and ritmo_ofensivo >= OFENSIVO_ABIERTO: tipo, desc = "PARTIDO ABIERTO E INTENSO",  "Alto ritmo en ambas fases con contacto frecuente"
    elif ritmo_ofensivo >= OFENSIVO_MUY_ABIERTO:                             tipo, desc = "PARTIDO OFENSIVO",           "Alto volumen de remates de ambos equipos"
    elif fisico_total < FISICO_NORMAL and ritmo_ofensivo < OFENSIVO_EQUILIBRADO: tipo, desc = "PARTIDO CONTROLADO",    "Bajo ritmo, busqueda de la solidez defensiva"
    else:                                                                    tipo, desc = "PARTIDO EQUILIBRADO",        "Sin tendencias dominantes claras entre ambos equipos"

    angles: list[str] = []
    if fouls_l > 14.0 or fouls_v > 14.0:                                 angles.append("alto_riesgo_disciplinario")
    if shots_l > 13.0 and shots_v > 13.0:                                angles.append("partido_abierto_bilateral")
    if dif_posesion > 10.0:                                               angles.append("asimetria_posesion")
    if fisico_total >= FISICO_MUY_ALTO:                                   angles.append("duelo_fisico_extremo")
    if shots_l < 9.0 and shots_v < 9.0:                                  angles.append("partido_cerrado_bajo_ritmo")
    if es_fis_l and es_tec_v:                                             angles.append("presion_vs_posesion_visitante")
    if es_tec_l and es_fis_v:                                             angles.append("posesion_local_vs_presion_visitante")

    return {
        "tipo_partido":      tipo,
        "tipo_partido_desc": desc,
        "estilos":           {"local": estilo_l, "visitante": estilo_v},
        "fisico_total":      round(fisico_total,   1),
        "fisico_label":      fisico_label,
        "ritmo_ofensivo":    round(ritmo_ofensivo, 1),
        "ofensivo_label":    ofensivo_label,
        "control_juego":     control_juego,
        "derived_angles":    angles,
    }


def calcular_xvolumen_eventos(
    xstyles: dict,
    equipo_local: str,
    equipo_visitante: str,
    contexto_comp: dict,
) -> dict:
    """Estima volumenes de eventos ajustados por factores del contexto competitivo."""
    sl = xstyles.get(equipo_local,     {})
    sv = xstyles.get(equipo_visitante, {})

    f_l = contexto_comp["local"]["factors"]["xg_factor"]
    f_v = contexto_comp["visitante"]["factors"]["xg_factor"]

    sh_l  = round(safe(sl.get("tiros"),          11.0) * f_l, 1)
    sh_v  = round(safe(sv.get("tiros"),          11.0) * f_v, 1)
    sot_l = round(safe(sl.get("tiros_a_puerta"),  4.0) * f_l, 1)
    sot_v = round(safe(sv.get("tiros_a_puerta"),  4.0) * f_v, 1)
    co_l  = round(safe(sl.get("corners"),         4.8) * f_l, 1)
    co_v  = round(safe(sv.get("corners"),         4.8) * f_v, 1)

    return {
        "shots_local":               sh_l,  "shots_visitante":               sh_v,  "shots_total":               round(sh_l + sh_v, 1),
        "shots_on_target_local":     sot_l, "shots_on_target_visitante":     sot_v, "shots_on_target_total":     round(sot_l + sot_v, 1),
        "corners_local":             co_l,  "corners_visitante":             co_v,  "corners_total":             round(co_l + co_v, 1),
        "offsides_total":            None,
    }


def generar_narrative(
    equipo_local: str,
    equipo_visitante: str,
    forma_local: dict,
    forma_visitante: dict,
    xgoals: dict,
    xposesion: dict,
    xtarjetas: dict,
    xfouls_result: dict,
    agresividad_vol: dict,
    contexto_comp: dict,
    compat: dict,
    contexto: dict,
) -> list[str]:
    """Genera bullets narrativos accionables para el knowledge pack."""
    xf_local = xfouls_result.get("xfouls_local",     xfouls_result.get("local",  0.0))
    xf_visit = xfouls_result.get("xfouls_visitante", xfouls_result.get("visitante", 0.0))
    xf_total = xfouls_result.get("xfouls_total",     xfouls_result.get("total",   0.0))

    def _forma_texto(equipo: str, forma: dict) -> str:
        if forma["partidos_analizados"] == 0:
            return f"{equipo}: sin datos de forma reciente."
        return (
            f"{equipo} acumula {forma['puntos']} pts en sus ultimos "
            f"{forma['partidos_analizados']} partidos "
            f"({forma['victorias']}V {forma['empates']}E {forma['derrotas']}D) "
            f"-- racha: {forma['racha_str']} -- tendencia: {forma['tendencia']}."
        )

    ccl = contexto_comp["local"]
    ccv = contexto_comp["visitante"]

    return [
        f"[FORMA LOCAL] {_forma_texto(equipo_local, forma_local)}",
        f"[FORMA VISITANTE] {_forma_texto(equipo_visitante, forma_visitante)}",
        (f"[GUION] Se espera un {compat['tipo_partido']}: {compat['tipo_partido_desc']}. "
         f"Intensidad fisica {compat['fisico_label']} · ritmo ofensivo {compat['ofensivo_label']} · "
         f"{compat['control_juego']}."),
        (f"[METRICAS] xG {xgoals['xg_local']}/{xgoals['xg_visitante']} (total {xgoals['xg_total']}) · "
         f"Posesion {xposesion['posesion_local']}%/{xposesion['posesion_visitante']}% · "
         f"xFaltas {xf_local}/{xf_visit} (total {xf_total}) · "
         f"Agresividad-volumen {agresividad_vol['local']:.1f}/{agresividad_vol['visitante']:.1f} "
         f"(total {agresividad_vol['total']:.1f}) · "
         f"xTarjetas {xtarjetas['xtarjetas_local']:.1f}/{xtarjetas['xtarjetas_visitante']:.1f} "
         f"(total {xtarjetas['xtarjetas_total']:.1f})."),
        (f"[PROBABILIDADES] Local {xgoals['prob_local_win']*100:.0f}% · "
         f"Empate {xgoals['prob_draw']*100:.0f}% · "
         f"Visitante {xgoals['prob_visitante_win']*100:.0f}% · "
         f"Over 2.5 {xgoals['prob_over25']*100:.0f}% · "
         f"BTTS {xgoals['prob_btts']*100:.0f}%."),
        (f"[CONTEXTO] Jornada ~{contexto['jornada_estimada']}/{contexto['jornadas_totales']} "
         f"({contexto['tramo_desc']}) · "
         f"{contexto['jornadas_restantes']} jornadas restantes · "
         f"presion de cierre: {contexto['presion_final']}."),
        (f"[CONTEXTO COMPETITIVO] {equipo_local}: ICC {ccl['scores']['icc']} ({ccl['lectura']}) · "
         f"{equipo_visitante}: ICC {ccv['scores']['icc']} ({ccv['lectura']})."),
        *(
            [f"[ANGULOS] Senales identificadas: {', '.join(compat['derived_angles'])}."]
            if compat["derived_angles"] else []
        ),
    ]
