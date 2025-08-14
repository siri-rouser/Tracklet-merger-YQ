import numpy as np
import logging
import cv2
import time
import json
from math import sqrt
from .config import TrackletMergerConfig
from typing import List, Dict, Tuple, Optional
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet,Trajectory,TrackletStatus,ZoneStatus

class SCTTrackbase:
    '''Keeps Trajectory() messages from the result of SCT'''
    def __init__(self,logger:logging,config: TrackletMergerConfig):
        self.data = Trajectory()
        self.logger = logger
        self.config = config
        self._zone_data_setup()
        self._process_flag = False

    def append(self, sae_msg: SaeMessage, stream_id: str):
        # Initialize the trajectory data
        if stream_id not in self.data.cameras:
            self.data.cameras[stream_id].CopyFrom(TrackletsByCamera())

        for track_id in sae_msg.trajectory.cameras[stream_id].tracklets:
            current_tracklet = sae_msg.trajectory.cameras[stream_id].tracklets[track_id]
            self.logger.debug(f'length of current_tracklet: {len(current_tracklet.detections_info)}')
            self.data.cameras[stream_id].tracklets[track_id].CopyFrom(current_tracklet)
            start_frame_id, end_frame_id = self._get_frame_ids(current_tracklet)
            
            self.data.cameras[stream_id].tracklets[track_id].start_frame = int(start_frame_id)
            self.data.cameras[stream_id].tracklets[track_id].end_frame = int(end_frame_id)
            self.data.cameras[stream_id].tracklets[track_id].end_time = time.time_ns()
            entry_zone_id, exit_zone_id, entry_zone_cls, exit_zone_cls = self._get_entry_exit_zones(current_tracklet, stream_id)
            self.data.cameras[stream_id].tracklets[track_id].zone.entry_zone_id = entry_zone_id
            self.data.cameras[stream_id].tracklets[track_id].zone.exit_zone_id = exit_zone_id
            self.data.cameras[stream_id].tracklets[track_id].zone.entry_zone_type = entry_zone_cls
            self.data.cameras[stream_id].tracklets[track_id].zone.exit_zone_type = exit_zone_cls
            self.data.cameras[stream_id].tracklets[track_id].status = self._default_status(entry_zone_cls, exit_zone_cls)
            self._process_flag = True

    def process(self,stream_id: str):
        if (stream_id not in self.data.cameras or len(self.data.cameras[stream_id].tracklets) < 2 or not self._process_flag):
            self.logger.debug(f"Skip processing for stream {stream_id}. Not enough tracklets or no new data.")
            return
        
        merged_any = True
        while merged_any:
            merged_any = False

            # Select candidate tracklets for merging
            candidate_ids: List[str] = [track_id for track_id, tracklet in self.data.cameras[stream_id].tracklets.items() if tracklet.status is TrackletStatus.SEARCHING and len(tracklet.detections_info) > 5]

            if len(candidate_ids) < 2:
                break

            # Save all candidate tracklets' data for processing
            start_frame: Dict[str, int] = {}
            end_frame: Dict[str, int] = {}
            entry_point: Dict[str, Tuple[float, float]] = {}
            exit_point: Dict[str, Tuple[float, float]] = {}
            start_feat: Dict[str, Optional[np.ndarray]] = {}
            end_feat: Dict[str, Optional[np.ndarray]] = {}

            for track_id in candidate_ids:
                current_tracklet = self.data.cameras[stream_id].tracklets[track_id]
                start_frame[track_id] = int(self.data.cameras[stream_id].tracklets[track_id].start_frame)
                end_frame[track_id] = int(self.data.cameras[stream_id].tracklets[track_id].end_frame)

                # Entry = earliest det, Exit = latest det (for pixel distance)
                first_det = min(current_tracklet.detections_info, key=lambda d: d.frame_id)
                last_det = max(current_tracklet.detections_info, key=lambda d: d.frame_id)
                entry_point[track_id] = self._center_abs(first_det, stream_id)
                exit_point[track_id] = self._center_abs(last_det, stream_id)

                # Features near start/end used for cosine distance
                # sfeat, _ = self._feature_near_start(current_tracklet)
                # efeat, _ = self._feature_near_end(current_tracklet)
                sfeat = current_tracklet.mean_feature
                efeat = current_tracklet.mean_feature
                start_feat[track_id] = sfeat
                end_feat[track_id] = efeat

            # Build candidate pairs with costs
            candidates: List[Tuple[float, str, str]] = []  # (cost, prev_id, next_id)
            self.logger.warning(f"[SCT merge] stream={stream_id} candidate_ids={candidate_ids}")
            for ida in candidate_ids:
                for idb in candidate_ids:
                    if ida == idb:
                        continue
                    # ensure temporal order a -> b
                    if end_frame[ida] >= start_frame[idb]:
                        continue
                    # temporal gap constraint
                    frame_gap = start_frame[idb] - end_frame[ida]
                    print(f"[SCT merge] stream={stream_id} {ida} -> {idb} | frame_gap={frame_gap}")
                    if frame_gap > self.config.sct_merging_config.max_frame_gap:
                        continue
                    # spatial proximity constraint
                    print(f"[SCT merge] stream={stream_id} {ida} -> {idb} | ecuclidean={self._euclidean(exit_point[ida], entry_point[idb]):.4f}")
                    if (self._euclidean(exit_point[ida], entry_point[idb]) > self.config.sct_merging_config.max_pixel_distance):
                        continue
                    # feature availability
                    fa = end_feat.get(ida)
                    fb = start_feat.get(idb)
                    if fa is None or fb is None:
                        self.logger.debug(f"[SCT merge] stream={stream_id} {ida} -> {idb} | features not available, skipping")
                        continue
                    self.logger.info(f"[SCT merge] stream={stream_id} {ida} -> {idb} | features cosine distance={self._cosine_distance(fa, fb):.4f}")
                    cost = self._cosine_distance(fa, fb)
                    if np.isfinite(cost) and cost < self.config.sct_merging_config.cosine_threshold:
                        candidates.append((float(cost), ida, idb))

            # Greedy solve: smallest cost first, avoid conflicts
            candidates.sort(key=lambda x: x[0])
            used_src: set = set()
            used_dst: set = set()
            planned_merges: List[Tuple[str, str, float]] = []
            for cost, a, b in candidates:
                if a in used_src or b in used_dst:
                    continue
                planned_merges.append((a, b, cost))
                used_src.add(a)
                used_dst.add(b)

            if not planned_merges:
                break

            # Apply merges
            for a, b, cost in planned_merges:
                if a not in self.data.cameras[stream_id].tracklets or b not in self.data.cameras[stream_id].tracklets:
                    continue
                # Merge b into a (a is the earlier tracklet)
                self._merge_tracklets(stream_id, a, b)
                self.logger.warning(
                    f"[SCT merge] stream={stream_id} {a} <- {b} | cost={cost:.4f}"
                )
                merged_any = True

        # After merging, we consider the processing handled
        self._process_flag = False

    def status_update(self, stream_id: str):
        current_time = time.time_ns()
        
        # Update the status of the tracklets in SCTTrackbase
        for track_id, tracklet in self.data.cameras[stream_id].tracklets.items():
            time_since_end = (current_time - tracklet.end_time) // 1_000_000  # in milliseconds
            if time_since_end > self.config.merging_config.searching_time and tracklet.status == TrackletStatus.ACTIVE:
                tracklet.status = TrackletStatus.SEARCHING
                self.logger.info(f"Tracklet {track_id} in stream {stream_id} is now searching, time since received: {time_since_end} ms")
            
            if time_since_end > self.config.merging_config.lost_time and tracklet.status == TrackletStatus.SEARCHING:
                tracklet.status = TrackletStatus.COMPLETED
                self.logger.info(f"Tracklet {track_id} in stream {stream_id} is now completed, time since received: {time_since_end} ms")

    def push_completed_tracklets(self, stream_id: str) -> Dict[str,Tracklet]:
        # Push completed tracklets to the MCTTrackbase and remove them from SCTTrackbase
        completed_tracklets = {}
        removed_tracklets = []
        if stream_id not in self.data.cameras:
            self.logger.warning(f"No tracklets found for stream {stream_id}.")
            return completed_tracklets
        
        for track_id, tracklet in self.data.cameras[stream_id].tracklets.items():
            if tracklet.status == TrackletStatus.COMPLETED:
                completed_tracklets[track_id]=tracklet
                self.logger.info(f"Pushing completed tracklet {track_id} from stream {stream_id} to MCTTrackbase.")
                # Remove the completed tracklet from SCTTrackbase
                removed_tracklets.append(track_id)

        for track_id in removed_tracklets:
            try:
                del self.data.cameras[stream_id].tracklets[track_id]
            except Exception as e:
                self.logger.warning(
                    f"Failed to delete completed tracklet {track_id} from stream {stream_id}: {e}"
                )
                
        return completed_tracklets

    def _center_abs(self, det, stream_id: str) -> Tuple[float, float]:
        bb = det.bounding_box
        # bounding boxes are normalized [0,1]; convert to pixels using config
        W, H = self.config.merging_config.original_img_size[stream_id]
        cx = 0.5 * (bb.min_x + bb.max_x) * W
        cy = 0.5 * (bb.min_y + bb.max_y) * H
        return (float(cx), float(cy))

    # def _feature_near_start(self, tracklet: Tracklet) -> Tuple[Optional[np.ndarray], Optional[object]]:
    #     """Earliest detection that has 'feature' populated."""
    #     dets = sorted(tracklet.detections_info, key=lambda d: d.frame_id)
    #     for d in dets:
    #         if hasattr(d, "feature") and len(d.feature) > 0:
    #             return np.asarray(list(d.feature), dtype=np.float32), d
    #     return None, None

    # def _feature_near_end(self, tracklet: Tracklet) -> Tuple[Optional[np.ndarray], Optional[object]]:
    #     """Latest detection that has 'feature' populated."""
    #     dets = sorted(tracklet.detections_info, key=lambda d: d.frame_id, reverse=True)
    #     for d in dets:
    #         if hasattr(d, "feature") and len(d.feature) > 0:
    #             return np.asarray(list(d.feature), dtype=np.float32), d
    #     return None, None

    @staticmethod
    def _cosine_distance(a, b) -> float:
        # Convert protobuf repeated float to numpy array if necessary
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.float32)
        else:
            a = a.astype(np.float32, copy=False)

        if not isinstance(b, np.ndarray):
            b = np.array(b, dtype=np.float32)
        else:
            b = b.astype(np.float32, copy=False)

        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return float("inf")

        sim = float(np.dot(a, b) / (na * nb))
        # numerical safety
        sim = max(min(sim, 1.0), -1.0)
        return 1.0 - sim  # cosine distance

    @staticmethod
    def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _merge_tracklets(self, stream_id: str, first_id: str, second_id: str):
        """Merge `second_id` into `first_id` (first ends before second starts)."""
        cam_store = self.data.cameras[stream_id]
        t1 = cam_store.tracklets[first_id]
        t2 = cam_store.tracklets[second_id]

        # Combine detections and sort by frame_id
        all_dets = list(t1.detections_info) + list(t2.detections_info)
        all_dets.sort(key=lambda d: d.frame_id)

        # Clear and re-add to t1 to keep the same key/ID
        # t1.detections_info.clear()  # <-- causes AttributeError with upb  
        t1.ClearField("detections_info")   # or: del t1.detections_info[:]

        for det in all_dets:
            new_det = t1.detections_info.add()
            new_det.CopyFrom(det)

        # Update start/end frames
        sf, ef = self._get_frame_ids(t1)
        t1.start_frame = int(sf)
        t1.end_frame = int(ef)
        t1.status = TrackletStatus.SEARCHING  # Reset status to ACTIVE after merging

        # Recompute zone info and status for the merged tracklet
        entry_zone_id, exit_zone_id, entry_zone_cls, exit_zone_cls = self._get_entry_exit_zones(t1, stream_id)
        t1.zone.entry_zone_id = entry_zone_id
        t1.zone.exit_zone_id = exit_zone_id
        t1.zone.entry_zone_type = entry_zone_cls
        t1.zone.exit_zone_type = exit_zone_cls
        t1.status = self._default_status(entry_zone_cls, exit_zone_cls)

        # refresh end_time
        t1.end_time = time.time_ns()

        # Remove the merged tracklet
        try:
            del cam_store.tracklets[second_id]
        except Exception as e:
            self.logger.warning(f"Failed to delete merged tracklet {second_id} from stream {stream_id}: {e}")


    def _get_frame_ids(self, tracklet: Tracklet):
        if not tracklet.detections_info:
            return 0,0 # No detections, return default frame IDs
        
        start_frame_id = min(detection.frame_id for detection in tracklet.detections_info)
        end_frame_id = max(detection.frame_id for detection in tracklet.detections_info)
        
        return start_frame_id, end_frame_id
    
    def _zone_data_setup(self):
        # Load zone data from the config
        self.zone_data = {}
        for stream_id, zone_path in self.config.merging_config.zone_data.items():
            if not zone_path.exists():
                self.logger.error(f"Zone data file {zone_path} does not exist.")
                continue
            
            with open(zone_path, 'r') as f:
                zone_info = json.load(f)
                self.zone_data[stream_id] = zone_info
                self.logger.info(f"Loaded zone data for {stream_id} from {zone_path}")
    
    def _get_entry_exit_zones(self, tracklet: Tracklet, stream_id: str):
        # Get entry and exit zones for the tracklet based on the zone data
        if not tracklet.detections_info:
            return -1, -1, ZoneStatus.UNDEFINED, ZoneStatus.UNDEFINED
        entry_zone_id = -1
        exit_zone_id = -1
        entry_zone_cls = ZoneStatus.UNDEFINED
        exit_zone_cls = ZoneStatus.UNDEFINED

        # Earliest detection
        entry_det = min(tracklet.detections_info, key=lambda d: d.frame_id)
        # Latest detection
        exit_det = max(tracklet.detections_info, key=lambda d: d.frame_id)

        def bbox_center_xyxy(bb):
            cx = 0.5 * (bb.min_x + bb.max_x) * self.config.merging_config.original_img_size[stream_id][0]
            cy = 0.5 * (bb.min_y + bb.max_y) * self.config.merging_config.original_img_size[stream_id][1]
            return (cx, cy)

        entry_point = bbox_center_xyxy(entry_det.bounding_box)
        exit_point  = bbox_center_xyxy(exit_det.bounding_box)

        def _is_point_in_bbox(point, bbox):
            x, y = point
            x1, y1, x2, y2 = bbox
            return x1 <= x <= x2 and y1 <= y <= y2

        for zone in self.zone_data[stream_id].values():
            if _is_point_in_bbox(entry_point,zone['rect_area']):
                entry_zone_id = int(zone['zone_id'])
                entry_zone_cls = ZoneStatus.ENTRY if zone['zone_cls'] == 'entry_zone' else (
                    ZoneStatus.EXIT if zone['zone_cls'] == 'exit_zone' else ZoneStatus.UNDEFINED)
            if _is_point_in_bbox(exit_point,zone['rect_area']):
                exit_zone_id = int(zone['zone_id'])
                exit_zone_cls = ZoneStatus.ENTRY if zone['zone_cls'] == 'entry_zone' else (
                    ZoneStatus.EXIT if zone['zone_cls'] == 'exit_zone' else ZoneStatus.UNDEFINED)
        
        return entry_zone_id, exit_zone_id, entry_zone_cls, exit_zone_cls
    
    def _default_status(self, entry_zone_cls, exit_zone_cls) -> TrackletStatus:
        if entry_zone_cls == ZoneStatus.ENTRY and exit_zone_cls == ZoneStatus.EXIT:
            return TrackletStatus.COMPLETED
        else:
            return TrackletStatus.ACTIVE