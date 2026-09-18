#!/usr/bin/env python3
"""Per-attribute (language / page-type) breakdown of an OmniDocBench eval.

Joins each category's per-page edit-distance (scorer/result/<prefix>_quick_match_<cat>_per_page_edit.json)
with the page's language + data_source from OmniDocBench.json, and reports mean edit per attribute.
Shows WHERE a model loses (which language / doc type), not just the global OVERALL.

  python3 tools/score_breakdown.py <scorer_result_prefix>
  e.g. tools/score_breakdown.py gravityocr_ar_full
"""
import json, os, sys
from collections import defaultdict

ROOT = os.environ.get("ROOT", "${DOCR_ROOT}")
RES = f"{ROOT}/scorer/result"
CATS = ["text_block", "display_formula", "table", "reading_order"]


def main():
    prefix = sys.argv[1]
    gt = json.load(open(f"{ROOT}/omnidocbench/OmniDocBench.json"))
    attr = {}  # basename -> (language, data_source)
    for p in gt:
        pi = p.get("page_info", {}); a = pi.get("page_attribute", {})
        bn = os.path.basename(pi.get("image_path", ""))
        attr[bn] = (a.get("language", "?"), a.get("data_source", "?"))

    # per category: img -> edit ; accumulate by language and by data_source
    by_lang = {c: defaultdict(list) for c in CATS}
    by_src = {c: defaultdict(list) for c in CATS}
    for c in CATS:
        f = f"{RES}/{prefix}_quick_match_{c}_per_page_edit.json"
        if not os.path.exists(f):
            print(f"  (missing {c})"); continue
        d = json.load(open(f))
        for img, e in d.items():
            if not isinstance(e, (int, float)):
                continue
            lang, src = attr.get(img, ("?", "?"))
            by_lang[c][lang].append(e); by_src[c][src].append(e)

    def tbl(title, acc):
        groups = sorted({g for c in CATS for g in acc[c]})
        print(f"\n=== {prefix} — edit-dist by {title} (lower better) ===")
        print(f"  {'group':22} {'n':>4} " + " ".join(f"{c[:9]:>9}" for c in CATS) + f" {'OVERALL':>8}")
        for g in groups:
            cells = []
            ov = []
            n = max((len(acc[c][g]) for c in CATS), default=0)
            for c in CATS:
                vals = acc[c][g]
                m = sum(vals)/len(vals) if vals else None
                cells.append(f"{m:9.4f}" if m is not None else f"{'·':>9}")
                if m is not None:
                    ov.append(m)
            ovm = f"{sum(ov)/len(ov):8.4f}" if len(ov) == 4 else f"{'·':>8}"
            print(f"  {g:22} {n:>4} " + " ".join(cells) + f" {ovm}")

    tbl("LANGUAGE", by_lang)
    tbl("PAGE-TYPE (data_source)", by_src)


if __name__ == "__main__":
    main()
