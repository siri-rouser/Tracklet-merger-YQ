#!/usr/bin/env python3
"""
Interpolate gaps in Multi-Cam Tracking TXT results.

Input format per line:
{cam_num} {gid} {frame_num} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {lat} {lon}

Example usage:
    python interpolate_tracklets.py --in input.txt --out output_interpolated.txt
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Row:
    cam: int
    gid: int
    frame: int
    x1: float
    y1: float
    w: float
    h: float
    lat: float
    lon: float

    def key(self) -> Tuple[int, int]:
        return (self.cam, self.gid)

def parse_line(line: str) -> Row | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) != 9:
        raise ValueError(f"Bad line (expect 9 columns): {line}")
    cam = int(parts[0])
    gid = int(parts[1])
    frame = int(parts[2])
    x1 = float(parts[3])*1.5
    y1 = float(parts[4])*1.5
    w  = float(parts[5])*1.5
    h  = float(parts[6])*1.5
    lat = float(parts[7])
    lon = float(parts[8])
    return Row(cam, gid, frame, x1, y1, w, h, lat, lon)

def row_to_str(r: Row) -> str:
    # Keep bbox to 2 decimals; lat/lon to high precision
    return f"{r.cam} {r.gid} {r.frame} {r.x1:.2f} {r.y1:.2f} {r.w:.2f} {r.h:.2f} {r.lat:.14f} {r.lon:.14f}"

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def interpolate_gap(a: Row, b: Row) -> List[Row]:
    """Interpolate strictly between frames a.frame and b.frame (exclusive)."""
    fa, fb = a.frame, b.frame
    gap = fb - fa
    if gap <= 1:
        return []
    out: List[Row] = []
    for f in range(fa + 1, fb):
        t = (f - fa) / (fb - fa)
        out.append(Row(
            cam=a.cam,
            gid=a.gid,
            frame=f,
            x1=lerp(a.x1, b.x1, t),
            y1=lerp(a.y1, b.y1, t),
            w= lerp(a.w,  b.w,  t),
            h= lerp(a.h,  b.h,  t),
            lat=lerp(a.lat, b.lat, t),
            lon=lerp(a.lon, b.lon, t),
        ))
    return out

def process(rows: List[Row]) -> List[Row]:
    # Group by (cam, gid)
    groups: Dict[Tuple[int,int], List[Row]] = {}
    for r in rows:
        groups.setdefault(r.key(), []).append(r)

    filled: List[Row] = []
    for key, lst in groups.items():
        # Sort by frame, and drop duplicate frames (keep first)
        lst.sort(key=lambda r: r.frame)
        dedup: List[Row] = []
        seen_frames = set()
        for r in lst:
            if r.frame not in seen_frames:
                dedup.append(r)
                seen_frames.add(r.frame)
        # Interpolate between consecutive observations
        all_rows = [dedup[0]] if dedup else []
        for i in range(len(dedup) - 1):
            a, b = dedup[i], dedup[i+1]
            all_rows.extend(interpolate_gap(a, b))
            all_rows.append(b)
        filled.extend(all_rows)

    # Sort globally for clean output
    filled.sort(key=lambda r: (r.cam, r.gid, r.frame))
    return filled

def main():
    ap = argparse.ArgumentParser(description="Interpolate missing frames within each (cam_num, gid) tracklet.")
    ap.add_argument("--in", dest="infile", required=True, help="Input TXT file")
    ap.add_argument("--out", dest="outfile", required=True, help="Output TXT file")
    args = ap.parse_args()

    # Read
    rows: List[Row] = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is not None:
                rows.append(parsed)
    print(f"Read {len(rows)} rows from {args.infile}")
    
    # Process
    result = process(rows)

    # Write
    with open(args.outfile, "w", encoding="utf-8") as f:
        for r in result:
            f.write(row_to_str(r) + "\n")

if __name__ == "__main__":
    main()
