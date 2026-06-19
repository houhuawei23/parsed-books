#!/usr/bin/env python3
"""Generate book index files (List.md) from metadata.yaml files.

Supports generating an index for a single directory, or for every category
subdirectory under a root directory.
"""

import argparse
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path
from collections import defaultdict


DEFAULT_OUTPUT = "List.md"


def discover_metadata(root: Path, recursive: bool = False):
    """Find all metadata.yaml files under subdirectories of root.

    By default only scans immediate subdirectories of root (depth=1).
    Set recursive=True to scan the entire subtree.
    """
    results = []
    if recursive:
        candidates = sorted(root.rglob("metadata.yaml"))
        for meta_file in candidates:
            results.append((meta_file.parent, meta_file))
    else:
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                meta_file = entry / "metadata.yaml"
                if meta_file.exists():
                    results.append((entry, meta_file))
    return results


def parse_date(value):
    """Return a display year string from various date formats."""
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return str(value.year)
    if hasattr(value, "year"):
        return str(value.year)
    if isinstance(value, str):
        match = re.search(r"\b(19|20)\d{2}\b", value)
        if match:
            return match.group(0)
    return str(value)


def normalize_authors(data):
    """Return a list of authors regardless of whether yaml has str or list."""
    authors = data.get("author") or data.get("authors", [])
    if isinstance(authors, str):
        return [authors]
    if isinstance(authors, list):
        return [str(a) for a in authors if a]
    return ["N/A"]


def normalize_keywords(data):
    """Return a clean list of keywords."""
    keywords = data.get("keywords", [])
    if isinstance(keywords, str):
        return [k.strip() for k in keywords.split(",") if k.strip()]
    if isinstance(keywords, list):
        return [str(k).strip() for k in keywords if k]
    return []


def load_metadata(meta_file: Path):
    with meta_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def escape_md(text: str) -> str:
    """Escape pipe characters inside markdown tables."""
    return str(text).replace("|", "\\|")


def anchor(text: str) -> str:
    """Create a GitHub-style markdown anchor from heading text."""
    anchor_text = re.sub(r"[^\w\s-]", "", text.lower())
    anchor_text = re.sub(r"[\s]+", "-", anchor_text.strip())
    return anchor_text


def build_index(book_dirs):
    """Build index entries from a list of (book_dir, meta_file) tuples."""
    entries = []
    keyword_index = defaultdict(list)

    for book_dir, meta_file in book_dirs:
        data = load_metadata(meta_file)
        title = data.get("title") or book_dir.name
        subtitle = data.get("subtitle", "")
        authors = normalize_authors(data)
        date = parse_date(data.get("date"))
        keywords = normalize_keywords(data)
        abstract = " ".join(data.get("abstract", "").splitlines()).strip()
        lang = data.get("lang", "")

        for kw in keywords:
            keyword_index[kw].append(title)

        entries.append({
            "dir": book_dir,
            "title": title,
            "subtitle": subtitle,
            "authors": authors,
            "date": date,
            "keywords": keywords,
            "abstract": abstract,
            "lang": lang,
        })

    return entries, keyword_index


def render_list(entries, keyword_index, title: str):
    lines = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> Auto-generated from `metadata.yaml`. Last updated: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    # Summary
    years = [int(e["date"]) for e in entries if e["date"].isdigit()]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total books**: {len(entries)}")
    if years:
        lines.append(f"- **Publication year range**: {min(years)} – {max(years)}")
    all_authors = sorted({a for e in entries for a in e["authors"]})
    if all_authors:
        lines.append(f"- **Authors**: {escape_md(', '.join(all_authors))}")
    all_langs = sorted({e["lang"] for e in entries if e["lang"]})
    if all_langs:
        lines.append(f"- **Languages**: {', '.join(all_langs)}")
    lines.append("")

    # Quick navigation
    lines.append("## Quick Navigation")
    lines.append("")
    for idx, e in enumerate(entries, 1):
        display = f"{e['title']}: {e['subtitle']}" if e["subtitle"] else e["title"]
        lines.append(f"{idx}. [{escape_md(display)}](#{idx}-{anchor(display)})")
    lines.append("")

    # Book details
    lines.append("## Books")
    lines.append("")

    for idx, e in enumerate(entries, 1):
        display_title = f"{e['title']}: {e['subtitle']}" if e["subtitle"] else e["title"]
        lines.append(f"### {idx}. {escape_md(display_title)}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Title** | {escape_md(e['title'])} |")
        if e["subtitle"]:
            lines.append(f"| **Subtitle** | {escape_md(e['subtitle'])} |")
        lines.append(f"| **Author(s)** | {escape_md(', '.join(e['authors']))} |")
        lines.append(f"| **Date** | {escape_md(e['date'])} |")
        lines.append(f"| **Directory** | [{escape_md(e['dir'].name)}]({e['dir'].name}) |")
        if e["lang"]:
            lines.append(f"| **Language** | {escape_md(e['lang'])} |")
        if e["keywords"]:
            lines.append(f"| **Keywords** | {escape_md(', '.join(e['keywords']))} |")
        lines.append("")
        lines.append("**Abstract**:")
        lines.append("")
        lines.append(e["abstract"] or "_No abstract provided._")
        lines.append("")

        # Useful subdirectories
        subdirs = [d for d in sorted(e["dir"].iterdir()) if d.is_dir() and not d.name.startswith(".")]
        if subdirs:
            subdir_links = [f"[{d.name}]({e['dir'].name}/{d.name})" for d in subdirs]
            lines.append("**Explore**: " + " · ".join(subdir_links))
            lines.append("")

        lines.append("---")
        lines.append("")

    # Keyword index
    if keyword_index:
        lines.append("## Keyword Index")
        lines.append("")
        for kw in sorted(keyword_index.keys(), key=lambda s: s.lower()):
            titles = keyword_index[kw]
            lines.append(f"- **{kw}**: {', '.join(escape_md(t) for t in titles)}")
        lines.append("")

    return "\n".join(lines)


def generate_for_directory(target_dir: Path, output_name: str, title: str,
                           recursive: bool, dry_run: bool, quiet: bool) -> int:
    """Generate a single index file for target_dir. Returns number of books indexed."""
    book_dirs = discover_metadata(target_dir, recursive=recursive)
    if not book_dirs:
        if not quiet:
            print(f"Skipping {target_dir}: no metadata.yaml found.")
        return 0

    entries, keyword_index = build_index(book_dirs)
    content = render_list(entries, keyword_index, title)
    output_path = target_dir / output_name

    if dry_run:
        print(f"[DRY-RUN] Would write {output_path} ({len(entries)} books)")
    else:
        output_path.write_text(content, encoding="utf-8")
        if not quiet:
            print(f"Generated {output_path} with {len(entries)} books.")
    return len(entries)


def discover_categories(root: Path):
    """Return immediate subdirectories of root that look like categories.

    A directory is considered a category if it contains at least one
    metadata.yaml file in its immediate subdirectories (i.e. book dirs).
    """
    categories = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            if discover_metadata(entry, recursive=False):
                categories.append(entry)
    return categories


def make_title(directory: Path, custom_title: str = None) -> str:
    """Derive a human-readable index title."""
    if custom_title:
        return custom_title
    name = directory.name
    return f"{name} Book Index"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown index files (List.md) from metadata.yaml files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # index every category under the script's root
  %(prog)s Causal                       # index ./Causal
  %(prog)s /path/to/Causal              # index an arbitrary directory
  %(prog)s --all-categories             # index every category under the root
  %(prog)s --all-categories --dry-run   # preview what would be generated
  %(prog)s -o README.md -t "My Library" # custom output filename and title
        """
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=None,
        help="Target directory to index. If omitted, the script's parent directory is used and all categories are indexed.",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output filename (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-t", "--title",
        default=None,
        help="Custom title for the index. Default is derived from the directory name.",
    )
    parser.add_argument(
        "-a", "--all-categories",
        action="store_true",
        help="Generate an index file for every category subdirectory of TARGET.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan TARGET for metadata.yaml files.",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )

    args = parser.parse_args()

    script_parent = Path(__file__).parent.resolve()
    if args.target is None:
        target = script_parent
        auto_all_categories = True
    else:
        target = args.target.resolve()
        auto_all_categories = False

    if not target.exists():
        print(f"Error: target directory does not exist: {target}", file=sys.stderr)
        sys.exit(1)
    if not target.is_dir():
        print(f"Error: target is not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    total_books = 0
    run_all_categories = args.all_categories or auto_all_categories

    if run_all_categories:
        categories = discover_categories(target)
        if not categories:
            if not args.quiet:
                print(f"No categories with metadata.yaml found under {target}.")
            sys.exit(0)

        for category_dir in categories:
            title = args.title or make_title(category_dir)
            total_books += generate_for_directory(
                category_dir,
                args.output,
                title,
                recursive=args.recursive,
                dry_run=args.dry_run,
                quiet=args.quiet,
            )
    else:
        title = args.title or make_title(target)
        total_books += generate_for_directory(
            target,
            args.output,
            title,
            recursive=args.recursive,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(f"Done. Total books indexed: {total_books}")


if __name__ == "__main__":
    main()
