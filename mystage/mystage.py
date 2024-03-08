import logging
from typing import Any, Dict, NamedTuple

from prometheus_client import Counter, Histogram, Summary
from visionapi.messages_pb2 import BoundingBox, SaeMessage

from .config import MyStageConfig

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('my_stage_get_duration', 'The time it takes to deserialize the proto until returning the tranformed result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
OBJECT_COUNTER = Counter('my_stage_object_counter', 'How many detections have been transformed')
PROTO_SERIALIZATION_DURATION = Summary('my_stage_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('my_stage_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class MyStage:
    def __init__(self, config: MyStageConfig) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)

    def __call__(self, input_proto) -> Any:
        return self.get(input_proto)
    
    @GET_DURATION.time()
    def get(self, input_proto):
        sae_msg = self._unpack_proto(input_proto)

        # Your implementation goes (mostly) here
        logger.warning('Received SAE message from pipeline')

        return self._pack_proto(sae_msg)
        
    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        return sae_msg
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _pack_proto(self, sae_msg: SaeMessage):
        return sae_msg.SerializeToString()