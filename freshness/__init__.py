"""
freshness — Stats freshness check for predecir-jornada-v2 Step 1.

Queries Supabase for the latest match date and compares to the earliest
user-provided match date. Gap > 14 days triggers a warning.
"""
