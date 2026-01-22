#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Tuple, Optional

def read_jsonl(path):
    """Yield (lineno, obj) for each valid JSON line; warn and skip invalid lines."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping line {i}: {e}", file=sys.stderr)

def write_jsonl(path, objs):
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def detect_id_key(example_obj: dict, preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    for k in ("new_global_id", "global_id", "globle_id", "new_id", "id"):
        if k in example_obj:
            return k
    raise KeyError(
        "Could not detect the global id field. Pass it explicitly with --id-key"
    )

def to_int_safe(v):
    """Convert v to int if possible; otherwise raise ValueError."""
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    # try parsing string
    return int(str(v))

def fix_ids(in_path: str, out_path: str, id_key: Optional[str] = None, dry_run: bool = False):
    # First pass: compute the minimum global id per (cam_id, ori_id)
    it = read_jsonl(in_path)
    try:
        first_lineno, first_obj = next(it)
    except StopIteration:
        print("[INFO] Input file is empty; nothing to do.", file=sys.stderr)
        if not dry_run:
            open(out_path, "w", encoding="utf-8").close()
        return {"total": 0, "changed": 0, "pairs": 0}

    detected_id_key = detect_id_key(first_obj, id_key)
    # restart iteration including the first line
    all_iter = [(first_lineno, first_obj)]
    all_iter.extend(list(it))

    min_by_pair: Dict[Tuple[str, str], int] = {}
    total = 0
    for lineno, obj in all_iter:
        total += 1
        cam = str(obj.get("cam_id"))
        ori = str(obj.get("ori_id"))
        if cam is None or ori is None:
            # Skip lines without required keys
            print(f"[WARN] Line {lineno} missing cam_id or ori_id; skipping for mapping.", file=sys.stderr)
            continue
        if detected_id_key not in obj:
            print(f"[WARN] Line {lineno} missing {detected_id_key}; skipping for mapping.", file=sys.stderr)
            continue
        try:
            gid = to_int_safe(obj[detected_id_key])
        except Exception as e:
            print(f"[WARN] Line {lineno} has non-integer {detected_id_key}={obj.get(detected_id_key)!r}; skipping for mapping.", file=sys.stderr)
            continue
        key = (cam, ori)
        if key not in min_by_pair or gid < min_by_pair[key]:
            min_by_pair[key] = gid

    # Second pass: write output with fixed ids
    changed = 0
    if not dry_run:
        out_objs = []
        for lineno, obj in all_iter:
            cam = str(obj.get("cam_id"))
            ori = str(obj.get("ori_id"))
            if cam is not None and ori is not None and detected_id_key in obj:
                try:
                    old_gid = to_int_safe(obj[detected_id_key])
                except Exception:
                    old_gid = obj[detected_id_key]
                new_gid = min_by_pair.get((cam, ori), old_gid)
                if old_gid != new_gid:
                    obj[detected_id_key] = new_gid
                    changed += 1
            out_objs.append(obj)
        write_jsonl(out_path, out_objs)

    report = {
        "total": total,
        "changed": changed,
        "pairs": len(min_by_pair),
        "id_key": detected_id_key,
        "input": in_path,
        "output": out_path,
    }
    return report

def main():
    ap = argparse.ArgumentParser(description="Unify global IDs per (cam_id, ori_id) pair by choosing the smallest ID.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL path")
    ap.add_argument("--id-key", dest="id_key", default=None, help="Field name for the global id (default: auto-detect)")
    ap.add_argument("--dry-run", action="store_true", help="Compute summary without writing output file")
    args = ap.parse_args()

    report = fix_ids(args.in_path, args.out_path, id_key=args.id_key, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()