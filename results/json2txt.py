#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Any, List

def to_int_flexible(val: Any):
    """
    Try to convert `val` to int. If it's a string with digits, extract the first integer portion.
    Otherwise return the original value.
    """
    try:
        return int(val)
    except Exception:
        if isinstance(val, str):
            m = re.search(r'-?\d+', val)
            if m:
                try:
                    return int(m.group(0))
                except Exception:
                    pass
        return val

def parse_bbox_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Ensure bbox is [x1, y1, x2, y2] and convert to xywh with x1/y1 being the top-left.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must be a list [x1,y1,x2,y2], got: {bbox}")
    x1, y1, x2, y2 = map(float, bbox)
    # Ensure proper top-left origin and positive width/height
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x_min, y_min, w, h

def main():
    ap = argparse.ArgumentParser(description="Convert JSONL detections to TXT lines: cam id frame x y w h lat lon")
    ap.add_argument("input_jsonl", type=Path, help="Path to input .jsonl")
    ap.add_argument("output_txt", type=Path, help="Path to output .txt")
    args = ap.parse_args()

    n_lines_in = 0
    n_lines_out = 0

    with args.input_jsonl.open("r", encoding="utf-8") as fin, args.output_txt.open("w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            n_lines_in += 1

            rec = json.loads(raw)

            cam_id = rec.get("cam_id")
            cam_num = to_int_flexible(cam_id)
            gid = rec.get("new_global_id")
            gid = to_int_flexible(gid)

            detections = rec.get("detections", []) or []
            for det in detections:
                bbox = det.get("bbox")
                if bbox is None:
                    continue

                x1, y1, w, h = parse_bbox_xyxy(bbox)

                # Frame number
                frame_num = det.get("frame_id", det.get("frame", det.get("frame_num")))
                frame_num = to_int_flexible(frame_num)

                # Geo coordinates: [lat, lon]
                geo = det.get("geo_coordinate") or det.get("geo") or det.get("wgs84")
                if isinstance(geo, (list, tuple)) and len(geo) >= 2:
                    try:
                        lat = float(geo[0])
                        lon = float(geo[1])
                    except Exception:
                        lat = -1
                        lon = -1
                else:
                    lat = -1
                    lon = -1

                # Format per user's spec (.2f for x/y/w/h; raw lat/lon as-is unless -1)
                if isinstance(lat, float) and isinstance(lon, float) and lat != -1 and lon != -1:
                    line = f"{cam_num} {gid} {frame_num} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {lat} {lon}\n"
                else:
                    line = f"{cam_num} {gid} {frame_num} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} -1 -1\n"

                fout.write(line)
                n_lines_out += 1

    print(f"Done. Read {n_lines_in} JSON records; wrote {n_lines_out} TXT lines to {args.output_txt}")
    
if __name__ == "__main__":
    main()