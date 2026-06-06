# Buzzword blocklist — terms that flag AI-generated academic text to reviewers
BUZZWORD_BLOCKLIST = [
    "delve", "delves", "delving", "delved",
    "tapestry", "tapestries",
    "underscores", "underscore",
    "it is worth noting", "it is important to note",
    "crucially", "importantly", "notably",
    "groundbreaking", "revolutionary", "game-changer", "game-changing",
    "paradigm shift", "paradigm-shifting",
    "cutting-edge", "state-of-the-art",
    "showcases", "showcase",
    "demonstrates the importance",
    "sheds light on", "shed light on",
    "paves the way", "pave the way",
    "in the realm of", "in the landscape of",
    "ever-evolving", "ever-changing",
    "holistic", "multifaceted", "nuanced approach",
    "leverage", "leverages", "leveraging",
    "synergy", "synergies", "synergistic",
    "robust", "comprehensive", "pivotal",
    "transformative", "transformational",
    "innovative solution", "novel approach",
    "it is evident that", "it is clear that",
    "needless to say",
]

EDITOR_SYSTEM_PROMPT = f"""\
You are a senior academic editor and writing specialist. Your role is to
polish and unify all sections of a PhD thesis chapter into publication-ready
academic prose.

## Banned Phrases
The following words and phrases are STRICTLY FORBIDDEN in your output.
Replace every occurrence with precise, formal academic language:
{chr(10).join(f"  - {w}" for w in BUZZWORD_BLOCKLIST)}

## Style Rules
- Write in third-person, formal academic register.
- Use hedging language appropriate to empirical claims
  (e.g., "the findings suggest", "evidence indicates", "it appears that").
- Prefer active voice for objectives; passive for methodology descriptions.
- Ensure all inline citations [Author, Year] are preserved exactly.
- Target reading level: academic journal (Flesch-Kincaid Grade 16+).
- Paragraphs: 4–6 sentences each; no orphan sentences.
- Do NOT add new content, facts, or citations — only refine prose.

## Output Format
Return a single JSON object:
{{
  "final_output": "<the complete, unified Introduction chapter in clean academic prose>"
}}

Structure the output as:
1. Background (refined)
2. Problem Statement (refined)
3. Research Questions
4. Research Objectives
5. Hypotheses
6. Scope and Delimitations
"""

EDITOR_USER_TEMPLATE = """\
Unify and refine the following draft thesis sections:

--- BACKGROUND ---
{background}

--- PROBLEM STATEMENT ---
{problem_statement}

--- RESEARCH QUESTIONS ---
{research_questions}

--- OBJECTIVES ---
{objectives}

--- HYPOTHESES ---
{hypotheses}

--- SCOPE ---
{scope}

Polish the prose and return the unified Introduction chapter now.
"""
