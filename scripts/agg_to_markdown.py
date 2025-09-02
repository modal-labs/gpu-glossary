import argparse
import os
import re
from functools import reduce
from pathlib import Path


def get_all_markdown_files(root_dir: str | Path) -> list[Path]:
    return list(Path(root_dir).rglob("*.md"))


def create_aggregated_markdown(root_dir: str | Path, links: bool) -> str:
    markdown_files = get_all_markdown_files(root_dir)

    processed_contents = list(map(lambda f: f.read_text(), markdown_files))
    return "\n\n".join(processed_contents)


def get_program_arguments() -> (str, str, bool):
    parser = argparse.ArgumentParser(description="Aggregate markdown files.")
    parser.add_argument(
        "--root",
        type=Path,
        default="gpu-glossary",
        help="Root directory to search for markdown files.",
    )
    parser.add_argument(
        "--output", type=Path, default="gpu-glossary.md", help="Output markdown file."
    )
    parser.add_argument(
        "--no-links", action="store_true", help="Remove markdown links."
    )
    args = parser.parse_args()
    return args.root, args.output, not args.no_links


def main() -> None:
    root_dir, output_file, links = get_program_arguments()
    aggregated_markdown = create_aggregated_markdown(root_dir, links)
    output_file.write_text(aggregated_markdown)


if __name__ == "__main__":
    main()
