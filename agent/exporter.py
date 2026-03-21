"""
Excel export.

Responsibilities:
- Write extracted call data to an Excel file.
- Schema: url, objective, inclusion_criteria, exclusion_criteria, deadline,
  max_funding, max_duration, procedure, contact, misc, remarks.
"""


def export_to_excel(records: list[dict], output_path: str) -> None:
    """
    Write a list of call records to an Excel file.

    Each record must contain the keys: url, objective, inclusion_criteria,
    exclusion_criteria, deadline, max_funding, max_duration, procedure,
    contact, misc, remarks.
    """
    raise NotImplementedError
