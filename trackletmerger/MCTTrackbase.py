import numpy as np
import logging
import cv2
import json
from math import sqrt
from pathlib import Path
from .config import TrackletMergerConfig
from sklearn.neighbors import KernelDensity
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet,Trajectory,TrackletStatus,ZoneStatus

class MCTTrackbase:
    def __init__(self,logger:logging,config: TrackletMergerConfig):
        self.data = Trajectory()
        self.logger = logger
        self.config = config
        self.last_processed_frame = 0
        self.results_file = Path(config.save_directory) / "cross_camera_matches.jsonl"
        self.frame_window = config.merging_config.frame_window  # e.g., 1000
        self.overlap_frames = config.merging_config.overlap_frames  # e.g., 200
        self._CLM_setup()

    def append(self, tracklets_dict:Dict[str,Tracklet], stream_id: str) -> None:
        """
        Append a new SaeMessage to the tracklet database.
        """
        if stream_id not in self.data.cameras:
            self.data.cameras[stream_id] = TrackletsByCamera()

        for track_id,tracklet in tracklets_dict.items():
            if track_id not in self.data.cameras[stream_id].tracklets:
                self.data.cameras[stream_id].tracklets[track_id] = tracklet
            else:
                self.logger.warning(f"Tracklet {track_id} already exists in stream {stream_id}. Skipping append.")

    def process(self, stream_id: str):
        # Process every 500-1000 frames, but with overlap
        current_frame = self._get_current_max_frame()
        
        if current_frame - self.last_processed_frame >= self.config.merging_config.frame_window:
            # Process tracklets from (last_processed_frame - overlap) to current_frame
            self.logger.info(f"Starting cross-camera processing at frame {current_frame}")
            
            start_frame = max(0, self.last_processed_frame)
            end_frame = current_frame

            candidate_tracklets = self._get_tracklets_in_range(start_frame, end_frame)

            if len(candidate_tracklets) < 2:
                self.logger.debug("Not enough tracklets for cross-camera matching")
                return

            reid_dict = self._perform_cross_camera_matching(candidate_tracklets, start_frame, end_frame)

            # Update processing state
            self.last_processed_frame = end_frame - self.overlap_frames

            self.logger.info(f"Completed processing window [{start_frame}, {end_frame}]")

    def _CLM_setup(self):
        '''
        Example Usage:
        time = np.array([[int(30)]])
        test_log_density = self.clm['1_to_2'][(13,2)].score_samples(time)[0]
        print('test_log_density',np.exp(test_log_density))
        '''
        self.clm = {}
        with open(self.config.merging_config.clm_path, "r") as f:
            cam_pair_data = json.load(f)

        for cam_a, to_dict in cam_pair_data.items():
            if not isinstance(to_dict, dict):
                continue

            for cam_b, info in to_dict.items():
                # Defensive checks
                if not isinstance(info, dict):
                    continue
                if "entry_exit_pair" not in info or "time_pair" not in info:
                    # Missing required fields; skip
                    continue

                key_str = f"{cam_a}_to_{cam_b}"
                entry_exit_pair = info["entry_exit_pair"]

                times = np.array(info["time_pair"])

                time_transition_data = times.reshape(-1, 1)

                kde = KernelDensity(kernel="gaussian", bandwidth=self.config.merging_config.clm_bandwidth)
                kde.fit(time_transition_data)
                
                if key_str not in self.clm:
                    self.clm[key_str] = {}

                self.clm[key_str][(entry_exit_pair[0],entry_exit_pair[1])] = kde
        
        self.logger.info(f"CLM setup complete with {len(self.clm)} camera pairs.")

    def _get_current_max_frame(self) -> int:
        """Get the maximum frame ID across all cameras."""
        max_frame = 0
        for camera_data in self.data.cameras.values():
            for tracklet in camera_data.tracklets.values():
                if tracklet.end_frame > max_frame:
                    max_frame = tracklet.end_frame
        return max_frame
    

    def _get_tracklets_in_range(self, start_frame: int, end_frame: int) -> Dict[str, List[Tuple[str, Tracklet]]]:
        """Get tracklets that overlap with the given frame range."""
        candidates = defaultdict(list)
        
        for stream_id, camera_data in self.data.cameras.items():
            for track_id, tracklet in camera_data.tracklets.items():
                # Check if tracklet overlaps with processing window
                if (tracklet.start_frame <= end_frame and tracklet.end_frame >= start_frame):
                    candidates[stream_id].append((track_id, tracklet))

        return candidates


    def _perform_cross_camera_matching(self, candidate_tracklets: Dict[str, List[Tuple[str, Tracklet]]], start_frame: int, end_frame: int):
        pass