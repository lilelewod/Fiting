"""Record a manual all-page visual-QA decision bound to one exact PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper/ieee_superquadric"


def parse_page_spec(spec: str) -> list[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 1 or last < first:
                raise ValueError(f"invalid page range: {part}")
            pages.update(range(first, last + 1))
        else:
            page = int(part)
            if page < 1:
                raise ValueError(f"invalid page number: {part}")
            pages.add(page)
    if not pages:
        raise ValueError("no visually confirmed pages were supplied")
    return sorted(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PAPER / "output/pdf/robust_parametric_surface_fitting.pdf",
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=PAPER / "tmp/pdfs/final_automated/rendered",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "output/pdf/visual_qa.json",
    )
    parser.add_argument(
        "--confirmed-pages",
        required=True,
        help="Pages manually inspected, e.g. 1-9 or 1,2,3. Must cover the full PDF.",
    )
    parser.add_argument(
        "--note",
        default="Checked for clipping, overlap, broken glyphs, table/figure legibility, and balanced final-page layout.",
    )
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    render_root = args.render_root.resolve()
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    page_count = len(PdfReader(str(pdf)).pages)
    confirmed_pages = parse_page_spec(args.confirmed_pages)
    expected_pages = list(range(1, page_count + 1))
    if confirmed_pages != expected_pages:
        raise ValueError(
            f"manual confirmation must cover exactly pages 1-{page_count}; got {confirmed_pages}"
        )

    rendered = sorted(render_root.glob("page-*.png"))
    if len(rendered) != page_count:
        raise ValueError(f"rendered-page mismatch: {len(rendered)} PNGs for {page_count} PDF pages")
    empty = [str(path) for path in rendered if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty rendered pages: {empty}")

    result = {
        "status": "PASS",
        "pdf": str(pdf),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_count": page_count,
        "rendered_page_count": len(rendered),
        "confirmed_pages": confirmed_pages,
        "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
