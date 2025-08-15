import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KernelDensity
import logging
from .config import TrackletMergerConfig
from matching_tool import CostMatrix, ReIDCalculator
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
                reid_dict,rm_dict = reid_cal.calc(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zn, q_exit_zn, g_enrty_zn, g_exit_zn,CLM=self.clm)
                self.logger.info(reid_dict)
                return reid_dict
            else:
                self.logger.info('dismat is empty')
                return None

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
