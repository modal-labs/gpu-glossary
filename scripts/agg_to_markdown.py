import os
import re
from functools import reduce
from typing import List, Tuple
import argparse


def get_all_markdown_files(root_dir: str) -> List[str]:
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.endswith('.md')
    ]


def read_file_content(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def create_aggregated_markdown(root_dir: str, links: bool) -> str:
    markdown_files = get_all_markdown_files(root_dir)
    
    def process_file(filepath: str) -> str:
        content = read_file_content(filepath)
        return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content) if not links else content
    
    processed_contents = list(map(process_file, markdown_files))
    aggregated_content = []
    for content in processed_contents:
        aggregated_content.append(content)
    return "\n".join(aggregated_content)


def get_program_arguments() -> Tuple[str, str, bool]:
    parser = argparse.ArgumentParser(description="Aggregate markdown files.")
    parser.add_argument('--root', type=str, default='gpu-glossary', help='Root directory to search for markdown files.')
    parser.add_argument('--output', type=str, default='gpu-glossary.md', help='Output markdown file.')
    parser.add_argument('--links', type=str, default='false', help='Keep markdown links (true/false).')
    args = parser.parse_args()
    links = args.links.lower() in ('true', '1', 'yes', 'on')
    return args.root, args.output, links


def main() -> None:
    root_dir, output_file, links = get_program_arguments()
    aggregated_content = create_aggregated_markdown(root_dir, links)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(aggregated_content)


if __name__ == "__main__":
    main()