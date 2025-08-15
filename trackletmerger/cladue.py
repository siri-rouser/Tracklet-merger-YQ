import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from visionapi_yq.messages_pb2 import Tracklet, TrackletStatus, ZoneStatus, Trajectory, TrackletsByCamera

class MCTTrackbase:
    def __init__(self, logger: logging, config):
        self.data = Trajectory()
        self.logger = logger
        self.config = config
        self._CLM_setup()
        
        # Memory management
        self.last_processed_frame = 0
        self.last_process_time = 0
        self.processed_matches = []  # Store finalized matches
        self.results_file = Path(config.output_dir if hasattr(config, 'output_dir') else "./results") / "cross_camera_matches.jsonl"
        
        # Processing windows
        self.frame_window = getattr(config.merging_config, 'frame_window', 1000)
        self.overlap_frames = getattr(config.merging_config, 'overlap_frames', 200) 
        self.process_interval = getattr(config.merging_config, 'process_interval', 10.0)
        
        # Initialize matched_dict similar to your original code
        self.matched_dict = {}
        if hasattr(config, 'input_stream_ids'):
            for stream_id in config.input_stream_ids:
                self.matched_dict[stream_id] = {}
        
        # Initialize mini_time
        self.mini_time = float('inf')

    def append(self, tracklets_dict: Dict[str, Tracklet], stream_id: str) -> None:
        """Append tracklets from SCT to MCT trackbase."""
        if stream_id not in self.data.cameras:
            self.data.cameras[stream_id] = TrackletsByCamera()

        for track_id, tracklet in tracklets_dict.items():
            if track_id not in self.data.cameras[stream_id].tracklets:
                self.data.cameras[stream_id].tracklets[track_id] = tracklet
                self.logger.info(f"Added tracklet {track_id} from stream {stream_id}")
            else:
                self.logger.warning(f"Tracklet {track_id} already exists in stream {stream_id}")

    def process(self, stream_id: str):
        """Main processing function with memory management."""
        current_time = time.time()
        current_max_frame = self._get_current_max_frame()
        
        # Check if we should process based on time or frame count
        should_process_time = (current_time - self.last_process_time) >= self.process_interval
        should_process_frame = (current_max_frame - self.last_processed_frame) >= self.frame_window
        
        if not (should_process_time or should_process_frame):
            return
            
        self.logger.info(f"Starting cross-camera processing at frame {current_max_frame}")
        
        # Define processing window with overlap
        start_frame = max(0, self.last_processed_frame - self.overlap_frames)
        end_frame = current_max_frame
        
        # Get tracklets in this frame range
        candidate_tracklets = self._get_tracklets_in_range(start_frame, end_frame)
        
        if len(candidate_tracklets) < 2:
            self.logger.debug("Not enough tracklets for cross-camera matching")
            return
        
        # Perform cross-camera matching
        matches = self._perform_cross_camera_matching(candidate_tracklets, start_frame, end_frame)
        
        # Save finalized matches and clean up memory
        self._finalize_and_save_matches(matches, start_frame, end_frame)
        
        # Update processing state
        self.last_processed_frame = end_frame - self.overlap_frames
        self.last_process_time = current_time
        
        self.logger.info(f"Completed processing window [{start_frame}, {end_frame}]")

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

    def _perform_cross_camera_matching(self, candidates: Dict[str, List], start_frame: int, end_frame: int) -> Dict:
        """Perform cross-camera matching using ReID approach similar to your existing code."""
        reid_dict = {}
        camera_pairs = self._get_camera_pairs()
        
        for cam_a, cam_b in camera_pairs:
            if cam_a not in candidates or cam_b not in candidates:
                continue
                
            # Convert to format similar to your existing matching_tool
            tracklets_a_dict = {track_id: tracklet for track_id, tracklet in candidates[cam_a]}
            tracklets_b_dict = {track_id: tracklet for track_id, tracklet in candidates[cam_b]}
            
            if not tracklets_a_dict or not tracklets_b_dict:
                continue
                
            # Create cost matrix using your existing approach
            cm = self._create_cost_matrix(tracklets_a_dict, tracklets_b_dict, cam_a, cam_b)
            if cm is None:
                continue
                
            dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_statuses, g_statuses, q_class_ids, g_class_ids = cm
            
            # Apply ReID calculation similar to your calc_reid function
            pair_reid_dict = self._calc_reid_matches(
                dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, 
                q_times, q_statuses, g_statuses, g_times, q_class_ids, g_class_ids,
                cam_a, cam_b
            )
            
            # Merge into main reid_dict
            for cam_id, tracks in pair_reid_dict.items():
                if cam_id not in reid_dict:
                    reid_dict[cam_id] = {}
                reid_dict[cam_id].update(tracks)
                
        return reid_dict

    def _create_cost_matrix(self, tracklets_a: Dict, tracklets_b: Dict, cam_a: str, cam_b: str):
        """Create cost matrix similar to your CostMatrix class."""
        try:
            # Extract features and metadata (similar to your _extract_tracklet_data)
            q_feats, q_track_ids, q_cam_ids, q_times, q_statuses, q_class_ids = self._extract_tracklet_data(tracklets_a, cam_a)
            g_feats, g_track_ids, g_cam_ids, g_times, g_statuses, g_class_ids = self._extract_tracklet_data(tracklets_b, cam_b)
            
            if q_feats is None or g_feats is None or q_feats.size(0) == 0 or g_feats.size(0) == 0:
                return None
                
            # Calculate distance matrix (using cosine distance by default)
            dismat = self._cosine_distance_matrix(q_feats, g_feats)
            
            return (dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, 
                   q_times, g_times, q_statuses, g_statuses, q_class_ids, g_class_ids)
                   
        except Exception as e:
            self.logger.error(f"Error creating cost matrix for {cam_a}->{cam_b}: {e}")
            return None

    def _extract_tracklet_data(self, tracklets: Dict, cam_id: str):
        """Extract features and metadata from tracklets."""
        import torch
        
        feats, track_ids, cam_ids, track_statuses, times, class_ids = [], [], [], [], [], []
        
        for track_id, tracklet in tracklets.items():
            if tracklet.mean_feature is None or len(tracklet.mean_feature) == 0:
                continue
                
            # Convert feature to tensor
            feat_tensor = torch.tensor(list(tracklet.mean_feature), dtype=torch.float).unsqueeze(0)
            feats.append(feat_tensor)
            track_ids.append(int(float(track_id)))
            cam_ids.append(cam_id)
            times.append([tracklet.start_frame, tracklet.end_frame])  # Using frame IDs instead of timestamps
            track_statuses.append(tracklet.status.name if hasattr(tracklet.status, 'name') else str(tracklet.status))
            
            # Get class ID from first detection
            if tracklet.detections_info and len(tracklet.detections_info) > 0:
                class_ids.append(tracklet.detections_info[0].object_id)
            else:
                class_ids.append(0)  # Default class
                
        if not feats:
            return None, None, None, None, None, None
            
        feats = torch.cat(feats, 0)
        track_ids = np.array(track_ids, dtype=np.int32)
        cam_ids = np.array(cam_ids)
        
        self.logger.debug(f'Extracted features for {cam_id}: {feats.size(0)}-by-{feats.size(1)} matrix')
        
        return feats, track_ids, cam_ids, times, track_statuses, class_ids

    def _cosine_distance_matrix(self, q_feats, g_feats):
        """Calculate cosine distance matrix."""
        import torch
        
        # Normalize features
        q_feats = torch.nn.functional.normalize(q_feats, p=2, dim=1)
        g_feats = torch.nn.functional.normalize(g_feats, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = torch.mm(q_feats, g_feats.t())
        
        # Convert to distance (1 - similarity)
        distmat = 1 - cosine_sim
        
        return distmat.numpy()

    def _calc_reid_matches(self, dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, 
                          q_times, q_statuses, g_statuses, g_times, q_class_ids, g_class_ids,
                          cam_a, cam_b):
        """ReID matching logic adapted from your calc_reid function."""
        reid_dict = {}
        indices = np.argsort(dismat, axis=1)
        
        # Configuration parameters (should be moved to config)
        dis_thre = self.config.merging_config.get('dis_thre', 0.7)
        dis_remove = self.config.merging_config.get('dis_remove', 0.9)
        dis_alpha = self.config.merging_config.get('dis_alpha', 0.8)
        dis_beta = self.config.merging_config.get('dis_beta', 0.2)
        
        for index, q_track_id in enumerate(q_track_ids):
            q_cam_id = q_cam_ids[index]
            q_class_id = q_class_ids[index]
            q_time = q_times[index]
            q_status = q_statuses[index]
            
            order = indices[index]
            
            # Status-based filtering
            status_remove = self._status_remove_gen(q_status, g_statuses, order)
            
            # Apply CLM/KDE filtering if available
            dismat = self._kde_filter(dismat, index, order, q_time, g_times, dis_alpha, dis_beta)
            
            # Combined removal criteria
            remove = ((g_cam_ids[order] == q_cam_id) | 
                     (g_class_ids[order] != q_class_id) |
                     (dismat[index][order] > dis_thre) |
                     status_remove)
            
            # Hard removal criteria  
            remove_hard = ((g_track_ids[order] == q_track_id) |
                          (g_cam_ids[order] == q_cam_id) |
                          (dismat[index][order] > dis_remove))
            
            keep_hard = np.invert(remove_hard)
            keep = np.invert(remove)
            
            if True not in keep_hard:
                continue
                
            # Get valid matches
            sel_g_dis = dismat[index][order][keep]
            sel_g_track_ids = g_track_ids[order][keep]
            sel_g_cam_ids = g_cam_ids[order][keep]
            
            if len(sel_g_dis) == 0:
                continue
                
            # Update reid_dict following your pattern
            min_dis = min(sel_g_dis)
            
            # First loop - for query tracklet
            if q_cam_id in reid_dict:
                if q_track_id in reid_dict[q_cam_id]:
                    if reid_dict[q_cam_id][q_track_id]["dis"] > min_dis:
                        reid_dict[q_cam_id][q_track_id]["dis"] = min_dis
                        reid_dict[q_cam_id][q_track_id]["id"] = q_track_id
                else:
                    reid_dict[q_cam_id][q_track_id] = {"dis": min_dis, "id": q_track_id}
            else:
                reid_dict[q_cam_id] = {q_track_id: {"dis": min_dis, "id": q_track_id}}
            
            # Second loop - for gallery tracklets
            for i in range(len(sel_g_track_ids)):
                g_track_id = sel_g_track_ids[i]
                g_cam_id = sel_g_cam_ids[i]
                g_dis = sel_g_dis[i]
                
                if g_cam_id in reid_dict:
                    if g_track_id in reid_dict[g_cam_id]:
                        if reid_dict[g_cam_id][g_track_id]["dis"] > g_dis:
                            reid_dict[g_cam_id][g_track_id]["dis"] = g_dis
                            reid_dict[g_cam_id][g_track_id]["id"] = q_track_id
                    else:
                        reid_dict[g_cam_id][g_track_id] = {"dis": g_dis, "id": q_track_id}
                else:
                    reid_dict[g_cam_id] = {g_track_id: {"dis": g_dis, "id": q_track_id}}
        
        return reid_dict

    def _status_remove_gen(self, q_status, g_statuses, order):
        """Status-based removal logic adapted from your code."""
        type_remove = []
        for ord in order:
            if q_status == 'COMPLETED':  # Equivalent to 'Lost'
                type_remove.append(True)
            elif q_status == 'ACTIVE':
                if g_statuses[ord] == 'SEARCHING':
                    type_remove.append(False)
                else:
                    type_remove.append(True)
            else:  # SEARCHING status
                if g_statuses[ord] == 'ACTIVE':
                    type_remove.append(False)
                else:
                    type_remove.append(True)
        return np.array(type_remove)

    def _kde_filter(self, dismat, index, orders, q_time, g_times, dis_alpha, dis_beta):
        """Apply KDE filtering using CLM if available."""
        for order in orders:
            # Calculate transition time (using frame IDs instead of timestamps)
            if q_time[0] - g_times[order][1] > 0:
                trans_time = q_time[0] - g_times[order][1]  # frames between end of g and start of q
            else:
                trans_time = g_times[order][0] - q_time[1]  # frames between end of q and start of g
            
            # Use CLM if available
            pdf = self._get_clm_probability(trans_time, orders, order)
            
            if pdf < 0.002:  # Low probability threshold
                dismat[index][order] = dismat[index][order] + 1  # Increase distance to filter out
            else:
                dismat[index][order] = dis_alpha * dismat[index][order] + (-1 * dis_beta * pdf)
                
        return dismat

    def _get_clm_probability(self, trans_time, orders, order):
        """Get CLM probability or use default."""
        try:
            # This would use your CLM model if available
            # For now, using a simple heuristic
            if abs(trans_time) > 300:  # More than 300 frames gap
                return 0.001
            elif abs(trans_time) < 50:  # Very close in time
                return 0.8
            else:
                return 0.5
        except:
            return 0.5

    def _get_camera_pairs(self) -> List[Tuple[str, str]]:
        """Get camera pairs for matching based on your camera topology."""
        camera_ids = sorted(self.data.cameras.keys())
        pairs = []
        
        # Create pairs based on camera connectivity
        # Adapt this to match your specific camera setup
        if len(camera_ids) >= 2:
            # For linear camera setup (1->2->3->4)
            for i in range(len(camera_ids) - 1):
                pairs.append((camera_ids[i], camera_ids[i + 1]))
                
        # You can add more complex topologies here if needed
        # For example, if cameras have different connection patterns
        
        self.logger.debug(f"Camera pairs for matching: {pairs}")
        return pairs 

    def _finalize_and_save_matches(self, reid_dict: Dict, start_frame: int, end_frame: int):
        """Save ReID matches and process results similar to your matching_result_process."""
        
        if not reid_dict:
            self.logger.debug("No matches to save")
            return
            
        # Process matches following your pattern
        self._process_reid_results(reid_dict, start_frame, end_frame)
        
        # Save to file in JSONL format
        self._save_reid_to_file(reid_dict, start_frame, end_frame)
        
        # Clean up old tracklets
        self._cleanup_old_tracklets(start_frame)

    def _process_reid_results(self, reid_dict: Dict, start_frame: int, end_frame: int):
        """Process ReID results similar to your matching_result_process function."""
        
        if not hasattr(self, 'matched_dict'):
            self.matched_dict = {}
            
        # Initialize matched_dict for each camera if not exists
        for stream_id in self.data.cameras.keys():
            if stream_id not in self.matched_dict:
                self.matched_dict[stream_id] = {}
        
        # Process reid_dict similar to your original logic
        for cam_id, track_matches in reid_dict.items():
            # Map camera ID to stream ID (adapt this to your naming convention)
            stream_key = self._map_cam_to_stream(cam_id)
            
            if stream_key not in self.matched_dict:
                continue
                
            for track_id, match_info in track_matches.items():
                track_key = str(float(track_id))
                
                # Initialize matched_dict entry if not exists
                if track_id not in self.matched_dict[stream_key]:
                    self.matched_dict[stream_key][track_id] = {
                        'ori_track_id': track_id,
                        'dis': match_info['dis'],
                        'new_track_id': match_info['id'],
                        'detections_info': []
                    }
                
                # Update with current tracklet information
                if track_key in self.data.cameras[stream_key].tracklets:
                    tracklet = self.data.cameras[stream_key].tracklets[track_key]
                    
                    # Update detection information
                    for detection_proto in tracklet.detections_info:
                        bbox = detection_proto.bounding_box
                        timestamp = detection_proto.timestamp_utc_ms if hasattr(detection_proto, 'timestamp_utc_ms') else detection_proto.frame_id
                        frame_id = detection_proto.frame_id
                        
                        detection_tuple = (bbox, timestamp, frame_id)
                        
                        if detection_tuple not in self.matched_dict[stream_key][track_id]['detections_info']:
                            self.matched_dict[stream_key][track_id]['detections_info'].append(detection_tuple)
                    
                    # Update distance with running average
                    current_detections = len(self.matched_dict[stream_key][track_id]['detections_info'])
                    if current_detections > 0:
                        old_dis = self.matched_dict[stream_key][track_id]['dis']
                        new_dis = match_info['dis']
                        self.matched_dict[stream_key][track_id]['dis'] = (
                            (1 / (current_detections + 1)) * new_dis + 
                            (current_detections / (current_detections + 1)) * old_dis
                        )
                else:
                    self.logger.warning(f"Track ID {track_id} not found in stream {stream_key}")
        
        # Update information for all active tracklets (similar to your second loop)
        for stream_key in self.matched_dict:
            if stream_key not in self.data.cameras:
                continue
                
            for track_id in list(self.matched_dict[stream_key].keys()):
                track_key = str(float(track_id))
                
                if (track_key in self.data.cameras[stream_key].tracklets and 
                    self.data.cameras[stream_key].tracklets[track_key].status == TrackletStatus.ACTIVE):
                    
                    tracklet = self.data.cameras[stream_key].tracklets[track_key]
                    for detection_proto in tracklet.detections_info:
                        bbox = detection_proto.bounding_box
                        timestamp = detection_proto.timestamp_utc_ms if hasattr(detection_proto, 'timestamp_utc_ms') else detection_proto.frame_id
                        frame_id = detection_proto.frame_id
                        
                        detection_tuple = (bbox, timestamp, frame_id)
                        
                        if detection_tuple not in self.matched_dict[stream_key][track_id]['detections_info']:
                            self.matched_dict[stream_key][track_id]['detections_info'].append(detection_tuple)

    def _map_cam_to_stream(self, cam_id: str) -> str:
        """Map camera ID to stream ID (adapt this to your naming convention)."""
        # This is a placeholder - adapt based on your camera/stream naming
        cam_to_stream_map = {
            'cam1': 'stream1',
            'cam2': 'stream2', 
            'cam3': 'stream3',
            'cam4': 'stream4',
            # Add more mappings as needed
        }
        return cam_to_stream_map.get(cam_id, cam_id)

    def _save_reid_to_file(self, reid_dict: Dict, start_frame: int, end_frame: int):
        """Save ReID results to file in JSONL format."""
        if not reid_dict:
            return
            
        # Create serializable version of reid_dict
        reid_record = {
            'timestamp': time.time(),
            'frame_range': [start_frame, end_frame],
            'reid_matches': {}
        }
        
        # Convert reid_dict to serializable format
        for cam_id, track_matches in reid_dict.items():
            reid_record['reid_matches'][cam_id] = {}
            for track_id, match_info in track_matches.items():
                reid_record['reid_matches'][cam_id][str(track_id)] = {
                    'distance': float(match_info['dis']),
                    'matched_id': int(match_info['id'])
                }
        
        # Append to results file
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(reid_record) + '\n')
            
        self.logger.info(f"Saved ReID results: {len(reid_dict)} cameras, frame range [{start_frame}, {end_frame}]")

    def get_matched_results(self) -> Dict:
        """Get current matched results (similar to your get_results method)."""
        return {
            'matched_dict': getattr(self, 'matched_dict', {}),
            'mini_time': getattr(self, 'mini_time', float('inf'))
        }

    def _cleanup_old_tracklets(self, processed_frame: int):
        """Remove tracklets that are fully processed and won't be needed."""
        cleanup_threshold = processed_frame - self.overlap_frames
        
        for stream_id, camera_data in self.data.cameras.items():
            tracklets_to_remove = []
            
            for track_id, tracklet in camera_data.tracklets.items():
                # Remove tracklets that ended before the cleanup threshold
                if tracklet.end_frame < cleanup_threshold and tracklet.status == TrackletStatus.COMPLETED:
                    tracklets_to_remove.append(track_id)
            
            for track_id in tracklets_to_remove:
                del camera_data.tracklets[track_id]
                self.logger.debug(f"Cleaned up tracklet {track_id} from {stream_id}")

    @staticmethod
    def _cosine_distance(feat_a, feat_b) -> float:
        """Calculate cosine distance between features."""
        if not isinstance(feat_a, np.ndarray):
            feat_a = np.array(feat_a, dtype=np.float32)
        if not isinstance(feat_b, np.ndarray):
            feat_b = np.array(feat_b, dtype=np.float32)
        
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance
        
        similarity = np.dot(feat_a, feat_b) / (norm_a * norm_b)
        return 1.0 - max(min(similarity, 1.0), -1.0)

    def _CLM_setup(self):
        """Setup Camera Link Model (existing implementation)."""
        self.clm = {}
        with open(self.config.merging_config.clm_path, "r") as f:
            cam_pair_data = json.load(f)

        for cam_a, to_dict in cam_pair_data.items():
            if not isinstance(to_dict, dict):
                continue

            for cam_b, info in to_dict.items():
                if not isinstance(info, dict):
                    continue
                if "entry_exit_pair" not in info or "time_pair" not in info:
                    continue

                key_str = f"{cam_a}_to_{cam_b}"
                entry_exit_pair = info["entry_exit_pair"]
                times = np.array(info["time_pair"])
                time_transition_data = times.reshape(-1, 1)

                from sklearn.neighbors import KernelDensity
                kde = KernelDensity(kernel="gaussian", bandwidth=self.config.merging_config.clm_bandwidth)
                kde.fit(time_transition_data)
                
                if key_str not in self.clm:
                    self.clm[key_str] = {}

                self.clm[key_str][(entry_exit_pair[0], entry_exit_pair[1])] = kde
        
        self.logger.info(f"CLM setup complete with {len(self.clm)} camera pairs.")

    def get_all_matches(self) -> List[Dict]:
        """Get all matches from file for final analysis."""
        matches = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    matches.append(json.loads(line.strip()))
        return matches