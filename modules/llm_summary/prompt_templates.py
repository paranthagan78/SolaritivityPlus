"""modules/llm_summary/prompt_templates.py"""

SUMMARY_PROMPT = """You are an expert solar PV panel analyst. Based on the detection and carbon emission data below, generate a concise professional report.

=== DETECTION DATA ===
{detection_data}

=== CARBON EMISSION DATA ===
{carbon_data}

Write a structured report with these sections:
1. Executive Summary (2-3 sentences)
2. Defect Analysis (what defects found, severity, frequency)
3. Carbon Impact Assessment (CO₂ implications, degradation impact)
4. Recommendations (maintenance actions, priority order)
5. Risk Level: [LOW / MEDIUM / HIGH / CRITICAL]

Be concise, factual, and actionable. Use plain text, no markdown."""