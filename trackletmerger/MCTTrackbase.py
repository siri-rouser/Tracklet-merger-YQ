import numpy as np
import logging
import cv2
import json
from math import sqrt
from .config import TrackletMergerConfig
from typing import List, Dict, Tuple, Optional
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet,Trajectory,TrackletStatus,ZoneStatus

class MCTTrackbase:
    def __init__(self,logger:logging,config: TrackletMergerConfig):
        self.data = Trajectory()
        self.logger = logger
        self.config = config

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