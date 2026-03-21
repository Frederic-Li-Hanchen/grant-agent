"""
Bekanntmachungen agent — main entrypoint.

Usage:
    python agent/agent.py --input mail.txt --output results.xlsx
"""
import argparse


def main(input_path: str, output_path: str) -> None:
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract key information from funding calls listed in a newsletter email.')
    parser.add_argument('--input', required=True, help='Path to the plain-text newsletter email file.')
    parser.add_argument('--output', required=True, help='Path for the output Excel file.')
    args = parser.parse_args()
    main(args.input, args.output)
