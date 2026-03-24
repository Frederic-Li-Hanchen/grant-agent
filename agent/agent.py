"""
Bekanntmachungen agent — main entrypoint.

Usage:
    python agent/agent.py --input mail.txt --output results.xlsx
"""
import argparse
import sys
import os
from time import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config_from_yaml
from fetcher import extract_call_links, fetch_call_text
from extractor import build_llm_and_embeddings, extract_fields
from exporter import export_to_excel


def main(input_path: str, output_path: str) -> None:

    start = time()

    # --- Load config ---
    config = load_config_from_yaml('config.yaml')

    # --- Read email text ---
    with open(input_path, 'r', encoding='utf-8') as f:
        email_text = f.read()

    # --- Identify call links ---
    urls = extract_call_links(email_text, config)
    print(f"Found {len(urls)} call link(s).")

    # --- Load LLM and embeddings once ---
    print("\nInitialising LLM and embeddings...")
    llm, embeddings, retry_on = build_llm_and_embeddings(config)
    print("Ready.\n")

    # --- Process each call ---
    records = []
    n_success = 0
    n_failed = 0

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")

        # Fetch call text
        call_text, remark = fetch_call_text(url, config)

        if not call_text:
            print(f"  FAILED: {remark}")
            records.append({'url': url, 'remarks': remark})
            n_failed += 1
            continue

        if remark:
            print(f"  WARNING: {remark}")

        # Extract fields via RAG
        try:
            fields = extract_fields(call_text, llm, embeddings, retry_on, config)
        except Exception as e:
            remark = f"{remark}; extraction error: {e}".lstrip('; ')
            print(f"  EXTRACTION FAILED: {e}")
            records.append({'url': url, 'remarks': remark})
            n_failed += 1
            continue

        fields['url'] = url
        fields['remarks'] = remark
        records.append(fields)
        n_success += 1
        print(f"  OK — {len(call_text)} chars, all fields extracted.")

    # --- Export to Excel ---
    export_to_excel(records, output_path)

    end = time()

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Extraction completed in {end - start:.2f} seconds.") 
    print(f"{len(urls)} link(s) found: {n_success} succeeded, {n_failed} failed.")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract key information from funding calls listed in a newsletter email.')
    parser.add_argument('--input', required=True,
                        help='Path to the plain-text newsletter email file.')
    parser.add_argument('--output', required=True,
                        help='Path for the output Excel file.')
    args = parser.parse_args()
    main(args.input, args.output)
