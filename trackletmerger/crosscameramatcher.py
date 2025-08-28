import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import KernelDensity
import logging
from .config import TrackletMergerConfig
from .matching_tool import CostMatrix, ReIDCalculator
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet,Trajectory,TrackletStatus,ZoneStatus

class CrossCameraMatcher:
    def __init__(self, logger:logging,config: TrackletMergerConfig):
        self.logger = logger
        self.config = config
        self.camera_pairs = config.merging_config.camera_pairs  # e.g., [['stream1', 'stream2'], ['stream2', 'stream3'], ['stream3', 'stream4']]
        if config.merging_config.is_clm:
            self._CLM_setup()
        else:
            self.clm = None

    def match(self, candidate_tracklets: Dict[str, List[Tuple[str, Tracklet]]], start_frame: int, end_frame: int):
        '''
        The reid_dict is a dictionary that contains the reid results for each camera pair.
        The structure is as follows:
        {
            int(cam_id): {
                int(track_id): {
                    "global_id": 1,
                    "local_matches": [{"matched_camera": int(cam_id), "matched_track_id": int(track_id), "distance": float(dis)}]
                }
            },

        '''


        if not hasattr(self, 'global_reid_dict'):
            self.global_reid_dict = {}
            self.global_id_counter = 0
            self.tracklet_to_global_id = {}

        for cam_a, cam_b in self.camera_pairs:
            # NOTE: in here, e.g. cam_a=stream1, cam_b=stream2
            if cam_a not in candidate_tracklets or cam_b not in candidate_tracklets:
                continue

            tracklets_b_dict:Dict[str, Tracklet] = {}
            tracklets_a_dict:Dict[str, Tracklet] = {}

            tracklets_a_dict = {track_id: tracklet for track_id, tracklet in candidate_tracklets[cam_a]}
            tracklets_b_dict = {track_id: tracklet for track_id, tracklet in candidate_tracklets[cam_b]}
            
            if not tracklets_a_dict or not tracklets_b_dict:
                continue

            cm = CostMatrix(tracklets1=tracklets_a_dict, tracklets2=tracklets_b_dict, cam1=cam_a, cam2=cam_b, logger=self.logger)
            dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zn, q_exit_zn, g_enrty_zn, g_exit_zn = cm.cost_matrix(metric=self.config.merging_config.matching_metric)

            reid_cal = ReIDCalculator(self.logger,dis_thre=self.config.merging_config.dis_thre, dis_remove=self.config.merging_config.dis_remove,
                                       dis_alpha=self.config.merging_config.dis_alpha,dis_beta=self.config.merging_config.dis_beta,kde_threshold=self.config.merging_config.kde_threshold,clm=self.clm)
            if dismat.size > 0:
                matches,rm_dict = reid_cal.calc(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zn, q_exit_zn, g_enrty_zn, g_exit_zn)
                self.logger.info(f"Camera pair {cam_a}-{cam_b} reid_dict: {matches}")

                self.global_reid_dict = self.merge_reid_dict(matches, (cam_a, cam_b))
            else:
                self.logger.info('dismat is empty')

        # Return the unified global reid dict
        self.logger.info(f"Final global reid dict:")
        self.logger.info(self.global_reid_dict)
        return self.global_reid_dict
        

    def _CLM_setup(self):
        '''
        Example Usage:
        time = np.array([[int(30)]])
        test_log_density = self.clm['1_to_2'].score_samples(time)[0]
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
                entry_exit_pair = info["entry_exit_pair"] # [(entry_exit_pair[0],entry_exit_pair[1])]

                times = np.array(info["time_pair"])

                time_transition_data = times.reshape(-1, 1)

                kde = KernelDensity(kernel="gaussian", bandwidth=self.config.merging_config.clm_bandwidth)
                kde.fit(time_transition_data)
                
                if key_str not in self.clm:
                    self.clm[key_str] = {}

                self.clm[key_str]['kde'] = kde
                self.clm[key_str]['entry_exit_pair'] = entry_exit_pair
        
        self.logger.info(f"CLM setup complete with {len(self.clm)} camera pairs.")

    def merge_reid_dict(self, matches:List, camera_pair: Tuple[str, str]) -> Dict[str, Dict[str, Dict]]:
        """
        Merge new reid_dict results with existing global reid_dict to maintain unified tracking IDs.
        
        Args:
            new_reid_dict: Reid results from current camera pair matching
            camera_pair: Current camera pair being processed (cam_a, cam_b)
        
        Returns:
            Updated unified reid_dict with consistent global IDs
        
        """
        if not hasattr(self, 'global_reid_dict'):
            self.global_reid_dict = {}
            self.global_id_counter = 0
            self.tracklet_to_global_id = {}  # Maps (cam_id, track_id) -> global_id
        
        if not matches:
            return self.global_reid_dict
        
        cam_a, cam_b = camera_pair
        
        
        # Process each match to assign global IDs
        for cam_id, track_id, matched_cam_id, matched_track_id, distance in matches:
            tracklet_key = (cam_id, track_id) 
            matched_tracklet_key = (matched_cam_id, matched_track_id)
            
            current_global_id = self.tracklet_to_global_id.get(tracklet_key)
            matched_global_id = self.tracklet_to_global_id.get(matched_tracklet_key)
            
            if current_global_id is None and matched_global_id is None:
                # Both tracklets are new, assign new global ID
                new_global_id = self._get_next_global_id()
                self.tracklet_to_global_id[tracklet_key] = new_global_id
                self.tracklet_to_global_id[matched_tracklet_key] = new_global_id
                self.logger.info(f"New global ID {new_global_id}: stream{cam_id}:{track_id} <-> stream{matched_cam_id}:{matched_track_id}")
                
            elif current_global_id is not None and matched_global_id is None:
                # Current tracklet has global ID, assign same to matched tracklet
                self.tracklet_to_global_id[matched_tracklet_key] = current_global_id
                self.logger.info(f"Extend global ID {current_global_id}: stream{matched_cam_id}:{matched_track_id} joins stream{cam_id}:{track_id}")
                
            elif current_global_id is None and matched_global_id is not None:
                # Matched tracklet has global ID, assign same to current tracklet
                self.tracklet_to_global_id[tracklet_key] = matched_global_id
                self.logger.info(f"Extend global ID {matched_global_id}: stream{cam_id}:{track_id} joins stream{matched_cam_id}:{matched_track_id}")
                
            elif current_global_id != matched_global_id:
                # Both have different global IDs, need to merge them
                # Keep the smaller ID and update all tracklets with the larger ID
                keep_id = min(current_global_id, matched_global_id)
                merge_id = max(current_global_id, matched_global_id)
                
                # Update all tracklets that had the merge_id to use keep_id
                for (c, t), g_id in list(self.tracklet_to_global_id.items()):
                    if g_id == merge_id:
                        self.tracklet_to_global_id[(c, t)] = keep_id
                
                self.logger.info(f"Merged global IDs: {merge_id} -> {keep_id} for stream{cam_id}:{track_id} <-> stream{matched_cam_id}:{matched_track_id}")
            
            # If both already have the same global ID, no action needed
        
        # Rebuild global_reid_dict from tracklet_to_global_id mapping
        self.global_reid_dict = {}
        for (cam_id, track_id), global_id in self.tracklet_to_global_id.items():
            if cam_id not in self.global_reid_dict:
                self.global_reid_dict[cam_id] = {}
            
            self.global_reid_dict[cam_id][track_id] = {
                "global_id": global_id,
                "local_matches": []  # Will be populated with match details
            }
        
        # Add match details to the global reid dict
        for cam_id, track_id, matched_cam_id, matched_track_id, distance in matches:
            if cam_id in self.global_reid_dict and track_id in self.global_reid_dict[cam_id]:
                match_detail = {
                    "matched_camera": matched_cam_id,
                    "matched_track_id": matched_track_id,
                    "distance": float(distance),
                    "camera_pair": camera_pair
                }
                self.global_reid_dict[cam_id][track_id]["local_matches"].append(match_detail)
        
        return self.global_reid_dict

    def _get_next_global_id(self) -> int:
        """Get the next available global ID."""
        self.global_id_counter += 1
        return self.global_id_counter

    def get_global_id_for_tracklet(self, cam_id: str, track_id: str) -> Optional[int]:
        """Get the global ID for a specific tracklet."""
        return self.tracklet_to_global_id.get((cam_id, track_id))

    def get_tracklets_with_global_id(self, global_id: int) -> List[Tuple[str, str]]:
        """Get all tracklets that share the same global ID."""
        return [(cam_id, track_id) for (cam_id, track_id), g_id in self.tracklet_to_global_id.items() 
                if g_id == global_id]

    def reset_global_tracking(self):
        """Reset global tracking state (useful for clean up memeory)."""
        self.global_reid_dict = {}
        # self.global_id_counter = 0
        self.tracklet_to_global_id = {}
        self.logger.info("Reset global tracking state")