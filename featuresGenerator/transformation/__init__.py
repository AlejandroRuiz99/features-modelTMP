"""Capa de transformacion del pipeline KDD.

Modulos de calculo estadistico puro (sin I/O):
  iap              — rankings de agresividad ponderada (IAP)
  xfouls           — faltas esperadas por partido
  xstyle           — perfil de juego por equipo
  referees         — perfiles estadisticos de arbitros
  referee_gmm      — perfiles GMM bimodales de arbitros
  competitive_context — contexto competitivo + fatiga (requiere laliga_objectives)
  market           — senales de mercado de apuestas
  xgoals           — xGoals y posesion esperada (Poisson bivariate)
  xposesion        — posesion esperada
  xtarjetas        — tarjetas esperadas y agresividad por volumen
  forma            — forma reciente y contexto de temporada
  match_profile    — perfil del partido (compatibilidad estilos, H2H, pace)
  knowledge_pack   — orquestador: ensambla el knowledge pack completo
"""

from transformation.iap import buscar_equipo, calcular_scores
from transformation.referees import buscar_arbitro, calcular_perfiles as calcular_perfiles_arbitros
from transformation.xfouls import calcular_xfouls
from transformation.xstyle import calcular_xstyle
from transformation.knowledge_pack import ensamblar_knowledge_pack
from transformation.market import ajustar_knowledge_pack
