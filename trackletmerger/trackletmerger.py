import logging
import time
import json
from typing import Any, Dict, NamedTuple,Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Summary
import sys
sys.path.append('../')
from visionapi_yq.messages_pb2 import SaeMessage
from visionlib.pipeline.tools import get_raw_frame_data

from .config import TrackletMergerConfig, LogLevel
from .utils import tracklet_match
from .trackletdatabase import Trackletdatabase
from .SCTTrackbase import SCTTrackbase
from .MCTTrackbase import MCTTrackbase

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('my_stage_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
PROTO_SERIALIZATION_DURATION = Summary('my_stage_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('my_stage_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class TrackletMerger:
    def __init__(self, config: TrackletMergerConfig, log_level: LogLevel) -> None:
        self._config = config
        logger.setLevel(log_level.value)
        self.sct_trackbase = SCTTrackbase(logger, config)
        self.mct_trackbase = MCTTrackbase(logger, config)
        self.last_processed_time = time.time_ns()

    def __call__(self, stream_id: str, input_proto: bytes = None) -> Any:
        return self.get(stream_id, input_proto)
    
    @GET_DURATION.time()
    def get(self, stream_id:str, input_proto:bytes = None) -> bytes:
        if input_proto is not None:
            sae_msg = self._unpack_proto(input_proto)
            current_time = time.time_ns()
            if len(sae_msg.trajectory.cameras[stream_id].tracklets) == 0 and (current_time - self.last_processed_time) // 1_000_000 < self._config.refresh_interval:
                return b''
        else:
            return b''

        inference_start = time.monotonic_ns()
        self.last_processed_time = time.time_ns()

        # put new sae_msg into SCTTrackbase
        if len(sae_msg.trajectory.cameras[stream_id].tracklets) > 0:
            self.sct_trackbase.append(sae_msg, stream_id)

        # update status for SCTTrackbase
        self.sct_trackbase.status_update(stream_id)

        # process the SCTTrackbase
        self.sct_trackbase.process(stream_id)

        # push completed scttracklets into MCTTrackbase
        tracklets_dict = self.sct_trackbase.push_completed_tracklets(stream_id)

        self.mct_trackbase.append(tracklets_dict, stream_id)

        # Greedy match from unmatched tracklets in MCTTrackbase (if there are any new tracklets in MCTTrackbase)
        self.mct_trackbase.process()

        # Save the matched results, prune the tracklets in MCTTrackbase
            
        inference_time_us = (time.monotonic_ns() - inference_start) // 1000

        return self._create_output(stream_id,inference_time_us,sae_msg)
        
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        # To use sae_msg: sae_msg.trajectory.cameras[camera_id].tracklets[track_id]
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        return sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _pack_proto(self, sae_msg: SaeMessage):
        return sae_msg.SerializeToString()
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, stream_id, inference_time_us, input_sae_msg:SaeMessage):
        out_sae_msg = SaeMessage()