import numpy as np
import logging
import threading
import cv2
from math import sqrt
from pathlib import Path
from .config import TrackletMergerConfig
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet,Trajectory,TrackletStatus,ZoneStatus
from .crosscameramatcher import CrossCameraMatcher
import json

class MCTTrackbase:
    def __init__(self,logger:logging,config: TrackletMergerConfig):
        self.data = Trajectory()
        self.logger = logger
        self.config = config
        self.last_processed_frame = 0

        self.results_file = Path(config.save_directory) / "cross_camera_matches.jsonl"
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        self.frame_window = config.merging_config.frame_window  # e.g., 1000
        self.overlap_frames = config.merging_config.overlap_frames  # e.g., 200
        self.matcher = CrossCameraMatcher(logger, config)
        self._state_lock = threading.Lock()   # protects self.data and frame markers
        self.stop_event = threading.Event()

    def _append(self, tracklets_dict:Dict[str,Tracklet], stream_id: str) -> None:
        """
        Append a new SaeMessage to the tracklet database.
        """
        if stream_id not in self.data.cameras:
            self.data.cameras[stream_id].CopyFrom(TrackletsByCamera())

        for track_id,tracklet in tracklets_dict.items():
            if track_id not in self.data.cameras[stream_id].tracklets:
                self.data.cameras[stream_id].tracklets[track_id].CopyFrom(tracklet)
            else:
                self.logger.warning(f"Tracklet {track_id} already exists in stream {stream_id}. Skipping append.")

    def process_async(self, tracklets_dict, stream_id, last_process_trigger) -> None:
        # 0) Append new tracklets under lock
        with self._state_lock:
            self._append(tracklets_dict, stream_id)

            current_frame = self._get_current_max_frame_locked()

            self.logger.info(f"Current max frame: {current_frame}, Last processed frame: {self.last_processed_frame} -------------------------")

            if ((current_frame - self.last_processed_frame) < self.config.merging_config.frame_window) and (not last_process_trigger):
                return
            
            start_frame = max(0, self.last_processed_frame)
            end_frame = current_frame
            # Snapshot candidates under lock; list/dicts are local copies
            candidate_tracklets = self._get_tracklets_in_range_locked(start_frame, end_frame)

        # Heavy work (no lock)
        if len(candidate_tracklets) < 2:
            self.logger.debug("Not enough tracklets for cross-camera matching")
            return

        self.logger.info(f"Starting cross-camera processing at frame {current_frame}")
        reid_dict = self.matcher.match(candidate_tracklets, start_frame, end_frame)
        self.matcher.reset_global_tracking()

        # Write results + cleanup under lock again (mutates self.data & markers)
        if reid_dict:
            self._process_results(reid_dict, candidate_tracklets, start_frame, end_frame)
        else:
            self.logger.info("No matches found in this processing window.")

        self.last_processed_frame = end_frame - self.overlap_frames

        with self._state_lock:
            self._cleanup_old_tracklets(start_frame - self.overlap_frames)

            self.logger.info(f"Completed processing window [{start_frame}, {end_frame}]")


    def _get_current_max_frame_locked(self) -> int:
        """Get the maximum frame ID across all cameras (lock must be held)."""
        max_frame = 0
        for camera_data in self.data.cameras.values():
            for tracklet in camera_data.tracklets.values():
                if tracklet.end_frame > max_frame:
                    max_frame = tracklet.end_frame
        return max_frame

    def _get_tracklets_in_range_locked(self, start_frame: int, end_frame: int) -> Dict[str, List[Tuple[str, Tracklet]]]:
        """Get tracklets that overlap with the given frame range (lock must be held)."""
        from collections import defaultdict
        candidates = defaultdict(list)
        for stream_id, camera_data in self.data.cameras.items():
            for track_id, tracklet in camera_data.tracklets.items():
                if (tracklet.start_frame <= end_frame and tracklet.end_frame >= start_frame):
                    candidates[stream_id].append((track_id, tracklet))
        return candidates

    def _process_results(
        self,
        reid_dict: Dict[int, Dict[int, Dict]],
        candidate_tracklets: Dict[str, List[Tuple[str, Tracklet]]],
        start_frame: int,
        end_frame: int,
    ):
        """
        Stream results to JSONL to reduce RAM usage.
        JSONL schema per line:
        {
            "processing_window": {"start_frame": int, "end_frame": int},
            "cam_id": str,
            "ori_id": str,
            "new_global_id": int,
            "detections": [
                {"bbox": [x1,y1,x2,y2], "frame_id": int, "geo_coordinate": [lat,lon]}
            ]
        }
        """
        if not reid_dict:
            self.logger.info(f"No matches found in window [{start_frame}, {end_frame}]")
            return

        # Build O(1) lookup: cam_id -> {track_id: Tracklet}
        tracklet_index: Dict[str, Dict[str, Tracklet]] = {}
        for cam_id, pairs in candidate_tracklets.items(): # Cam ID is string in here.
            idx = {}
            for tid, trk in pairs:
                idx[int(tid)] = trk # tid is string, trk is Tracklet
            tracklet_index[int(cam_id.replace('stream',''))] = idx # make str to int

        # Ensure output directory exists
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        wrote = 0
        with open(self.results_file, "a", encoding="utf-8") as f:
            for cam_id, track_matches in reid_dict.items():
                cam_id = int(cam_id)  
                # Resolve original image size once per camera
                try:
                    W, H = self.config.merging_config.original_img_size[f"stream{cam_id}"]
                except Exception:
                    # Fallback to a safe default (adjust if you prefer to raise)
                    W, H = 2560, 1440

                for track_id, match_info in track_matches.items():
                    track_id = int(track_id)  # Ensure track_id is an integer
                    global_id = match_info.get("global_id")
                    if global_id is None:
                        continue

                    tracklet = tracklet_index.get(cam_id, {}).get(track_id)
                    if tracklet is None:
                        self.logger.warning(f"Could not find tracklet {cam_id}:{track_id}")
                        continue

                    detections = []
                    # Build detection records (avoid unnecessary temporaries)
                    for det in tracklet.detections_info:
                        bbox = det.bounding_box
                        x1 = max(bbox.min_x * W, 0)
                        y1 = max(bbox.min_y * H, 0)
                        x2 = min(bbox.max_x * W, W)
                        y2 = min(bbox.max_y * H, H)
                        # Handle missing geo_coordinate safely
                        lat, lon = -1.0, -1.0
                        try:
                            # For proto3 messages, HasField works for message fields
                            if det.HasField("geo_coordinate"):
                                lat = float(det.geo_coordinate.latitude)
                                lon = float(det.geo_coordinate.longitude)
                        except Exception:
                            # If HasField isn't available, fall back to attribute checks
                            gc = getattr(det, "geo_coordinate", None)
                            if gc is not None:
                                lat = float(getattr(gc, "latitude", -1.0))
                                lon = float(getattr(gc, "longitude", -1.0))

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "frame_id": int(det.frame_id),
                            "geo_coordinate": [lat, lon],
                        })

                    record = {
                        "processing_window": {"start_frame": int(start_frame), "end_frame": int(end_frame)},
                        "cam_id": cam_id,
                        "ori_id": track_id,
                        "new_global_id": int(global_id),
                        "detections": detections,
                    }

                    f.write(json.dumps(record) + "\n")
                    wrote += 1

                    self.logger.info(
                        f"Saved: {cam_id}:{track_id} -> Global ID {global_id} "
                        f"({len(detections)} detections)"
                    )

        if wrote == 0:
            self.logger.info("No tracklet results to save")
        else:
            self.logger.info(f"Saved {wrote} tracklet results to {self.results_file}")


    def _cleanup_old_tracklets(self, cutoff_frame: int):
        """
        Remove tracklets that are older than cutoff_frame to minimize memory usage.
        Keep only tracklets that might be needed for future processing windows.
        """
        if cutoff_frame <= 0:
            return
            
        removed_count = 0
        for stream_id in list(self.data.cameras.keys()):
            tracklets_to_remove = []
            
            for track_id, tracklet in self.data.cameras[stream_id].tracklets.items():
                # Remove tracklets that ended before the cutoff and are completed
                if tracklet.end_frame < cutoff_frame and tracklet.status == TrackletStatus.COMPLETED:
                    tracklets_to_remove.append(track_id)
            
            for track_id in tracklets_to_remove:
                del self.data.cameras[stream_id].tracklets[track_id]
                removed_count += 1

        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old tracklets from memory (cutoff_frame: {cutoff_frame})")