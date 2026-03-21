"""
Excel export.

Responsibilities:
- Write extracted call data to an Excel file.
- Schema: url, objective, inclusion_criteria, exclusion_criteria, deadline,
  max_funding, max_duration, procedure, contact, misc, remarks.
"""
import pandas as pd


COLUMNS = [
    'url',
    'objective',
    'inclusion_criteria',
    'exclusion_criteria',
    'deadline',
    'max_funding',
    'max_duration',
    'procedure',
    'contact',
    'misc',
    'remarks',
]


def export_to_excel(records: list[dict], output_path: str) -> None:
    """
    Write a list of call records to an Excel file.

    Each record should contain the keys defined in COLUMNS. Missing keys are
    filled with an empty string. Extra keys are ignored.
    """
    rows = [{col: record.get(col, '') for col in COLUMNS} for record in records]
    df = pd.DataFrame(rows, columns=COLUMNS)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Bekanntmachungen')

        ws = writer.sheets['Bekanntmachungen']

        # Auto-size each column to fit its content
        for col_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col_cells
            )
            # Cap width and add a small margin; Excel column width unit ≈ 1 char
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 60)

        # Wrap text and set a fixed row height for data rows so long fields remain readable
        from openpyxl.styles import Alignment
        for row in ws.iter_rows(min_row=2):
            ws.row_dimensions[row[0].row].height = 80
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

        # Bold header row
        from openpyxl.styles import Font
        for cell in ws[1]:
            cell.font = Font(bold=True)
