"""
RAG-based information extraction.

Responsibilities:
- Preprocess call text (formatting, cleaning, chunking).
- Run vector-based RAG to retrieve each target field.
- Support ministry-specific prompt templates via config.yaml.
"""


def extract_fields(call_text: str, ministry: str | None = None) -> dict:
    """
    Extract target fields from a call document using vector RAG.

    Args:
        call_text: Plain text of the funding call.
        ministry: Optional ministry/domain identifier for prompt selection.

    Returns dict with keys: objective, inclusion_criteria, exclusion_criteria,
    deadline, max_funding, max_duration, procedure, contact, misc.
    """
    raise NotImplementedError
