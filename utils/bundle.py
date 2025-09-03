# Bundles the GPU Glossary contents into a single Markdown file for distribution.
import argparse
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# We have two extra dependencies: `cairosvg` and `python-frontmatter`.
# `cairosvg` requires system packages. See their docs for instructions on your platform.
import cairosvg
import frontmatter


here = Path(__file__).parent
ROOT = here.parent


def main() -> None:
    args = get_program_arguments()
    aggregated_markdown = aggregate_contents(args.contents_json_path, convert_svgs=args.convert_svgs_to_pngs)
    args.output_file.write_text(aggregated_markdown)
    print(args.output_file)


def aggregate_contents(contents_path: str | Path, convert_svgs=False) -> str:
    pages = load_from_contents_path(Path(contents_path))
    processed_contents = list(map(functools.partial(render_page, convert_svgs=convert_svgs), pages))
    return "# GPU Glossary\n\n" + "\n\n".join(processed_contents)


@dataclass
class Page:
    """A Page in the GPU Glossary."""
    title: str
    href: str
    abbreviation: str | None
    pages: list["Page"] = field(default_factory=list)


    @staticmethod
    def from_dict(data: dict) -> "Page":
        return Page(
            title=data["title"],
            href=data["href"],
            abbreviation=data.get("abbreviation"),
            pages=[Page.from_dict(p) for p in data.get("pages", [])]
        )

    @staticmethod
    def from_json(data: str) -> list["Page"]:
        raw = json.loads(data)
        return [Page.from_dict(item) for item in raw]


def render_page(page: Page, level=0, convert_svgs=False) -> str:
    """Convert a Page into a string, including its subpages."""
    print(f"Rendering {page.title}")
    content = f"##{'#' * min(level, 2)} {page.title}\n\n"
    path = (Path(".") / page.href.lstrip("/")).with_suffix(".md")
    post = frontmatter.load(path)
    content += post.content
    content = re.sub(r"/gpu-glossary/(?:[^/]+/)?([^/\s]+)", r"#\1", content)
    content = content.replace(
        "(themed-image://",
        "(https://modal-cdn.com/gpu-glossary/light-"
    )
    if convert_svgs:
        content = replace_svgs_with_pngs(content)
    content += "\n\n"
    content += "\n\n".join(render_page(page, level=level + 1, convert_svgs=convert_svgs) for page in page.pages)
    return content.strip()


def replace_svgs_with_pngs(text: str, out_dir: Path = ROOT / "dist" / "diagrams") -> str:
    """Replace all GPU Glossary diagram SVGs with PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def repl(match: re.Match) -> str:
        svg_url = match.group(0)
        stem = Path(svg_url).stem
        png_path = out_dir / f"{stem}.png"

        if not png_path.exists():
            cairosvg.svg2png(url=svg_url, write_to=str(png_path))

        return str(png_path.relative_to(ROOT))

    pattern = r"https://modal-cdn\.com/gpu-glossary/[\w\-.]+\.svg"
    return re.sub(pattern, repl, text)


def load_from_contents_path(contents_path: Path) -> list[Page]:
    pages = Page.from_json(contents_path.read_text())[0].pages
    return pages


def get_program_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate markdown files.")
    parser.add_argument(
        "--contents-json-path",
        type=Path,
        default= ROOT / "dist" / "contents.json",
        help="Path to a JSON-formatted table of contents",
    )
    parser.add_argument("--convert-svgs-to-pngs", action="store_true", help="Convert all SVGs from the Modal CDN to PNGs.")
    parser.add_argument(
        "--output-file", type=Path, default=ROOT / "dist" / "gpu-glossary.md", help="Output markdown file."
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
