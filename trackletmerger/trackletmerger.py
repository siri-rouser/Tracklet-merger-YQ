import logging
from typing import Any, Dict, NamedTuple,Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Summary
import sys
sys.path.append('../')
from visionapi_yq.messages_pb2 import BoundingBox, SaeMessage
from visionlib.pipeline.tools import get_raw_frame_data

from .buffer import MessageBuffer
from .config import MergingConfig, LogLevel
from .utils import tracklet_match
from .trackletdatabase import Trackletdatabase

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('my_stage_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('my_stage_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('my_stage_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('my_stage_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class TrackletMerger:
    def __init__(self, config: MergingConfig, log_level: LogLevel) -> None:
        self._config = config
        # self._buffer = MessageBuffer()

        logger.setLevel(log_level.value)
        self._trackletdatabase = Trackletdatabase(logger)

    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
    
    @GET_DURATION.time()
    def get(self, stream_id, input_proto:bytes = None) -> List[Tuple[str, bytes]] :
        sae_msg = None
        if input_proto is not None:
            frame_image,sae_msg = self._unpack_proto(input_proto)

        else:
            return [(self._config.output_stream_id, b'')]
        
        logger.debug(f"[DEBUG] Unpacking SAE message, timestamp: {sae_msg.frame.timestamp_utc_ms}")

        self._trackletdatabase.append(frame_image,stream_id,sae_msg)
        self._trackletdatabase.tracklet_status_update(stream_id,sae_msg)

        # tracklet_match(sae_msg)
        # NOTE: change a data source to test again
        if (self._trackletdatabase.data.cameras['stream1'].tracklets is not None) and (self._trackletdatabase.data.cameras['stream2'].tracklets is not None):
            tracklets1 = self._trackletdatabase.data.cameras['stream1'].tracklets
            tracklets2 = self._trackletdatabase.data.cameras['stream2'].tracklets
            reid_dict = tracklet_match(logger,frame_image,tracklets1,tracklets2)

        # self._trackletdatabase.matching_result_process(stream_id,reid_dict,sae_msg)
        self._trackletdatabase.prune(stream_id)
            
        # Your implementation goes (mostly) here
        logger.warning('Received SAE message from pipeline')
        out_msg = sae_msg
        # return [(self._config.output_stream_id, self._pack_proto(out_msg))]
        return [(self._config.output_stream_id, self._pack_proto(out_msg))]
        
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        input_frame = sae_msg.frame
        input_image = get_raw_frame_data(input_frame)

        return input_image, sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _pack_proto(self, sae_msg: SaeMessage):
        return sae_msg.SerializeToString()
    