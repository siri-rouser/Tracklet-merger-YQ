import logging
import time
import json
import threading
import queue
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
        self.last_process_trigger = False
        logger.setLevel(log_level.value)
        self.sct_trackbase = SCTTrackbase(logger, config)
        self.mct_trackbase = MCTTrackbase(logger, config)
        self._state_lock = threading.Lock() 
        self.last_processed_time = time.time_ns()
        self._sct_queues: dict[str, queue.Queue] = {}
        self._sct_threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()

    def __call__(self, stream_id: str, input_proto: bytes = None) -> Any:
        return self.get(stream_id, input_proto)
    
    def _ensure_worker(self, stream_id: str, max_qsize: int = 64) -> None:
        if stream_id in self._sct_threads:
            return
        q = queue.Queue(maxsize=max_qsize)
        self._sct_queues[stream_id] = q
        t = threading.Thread(target=self._stream_worker, args=(stream_id,), daemon=True)
        t.start()
        self._sct_threads[stream_id] = t
        logger.info(f"SCT worker started for {stream_id}")

    # NEW: the worker runs your two lines (SCT, then MCT)
    def _stream_worker(self, stream_id: str):
        q = self._sct_queues[stream_id]
        while not self._stop_event.is_set():
            try:
                sae_msg = q.get(timeout=0.5) # Get the next message from the queue, will wait 0.5 seconds if there is nothing in the queue, otherwise, get the message immediately
            except queue.Empty:
                continue

            try:
                # Single-Camera processing
                tracklets_dict = self.sct_trackbase.sct_process(sae_msg, stream_id)

                # Cross-camera processing (thread-safe; see MCT patch below)
                self.mct_trackbase.process_async(tracklets_dict, stream_id, self.last_process_trigger)

                if self.last_process_trigger:
                    self._stop_event.set()

            except Exception as e:
                logger.exception(f"SCT/MCT worker error on {stream_id}: {e}")
            finally:
                q.task_done() # tell the queue we're done with that item



    # (optional) call this on shutdown
    def stop(self):
        self._stop_event.set()
    
    @GET_DURATION.time()
    def get(self, stream_id:str, input_proto:bytes = None) -> bytes:
        current_time = time.time_ns()
        if input_proto is not None:
            sae_msg = self._unpack_proto(input_proto)
            if len(sae_msg.trajectory.cameras[stream_id].tracklets) == 0 and ((current_time - self.last_processed_time) // 1_000_000 < self._config.refresh_interval):
                return b''
        else:
            if (current_time - self.last_processed_time) // 1_000_000 < self._config.last_process_interval:
                return b''
            else:
                logger.warning('The last processing, program exit after processing ...')
                self.last_process_trigger = True
                sae_msg = SaeMessage()

        self.last_processed_time = time.time_ns()

        # ENSURE worker and enqueue quickly (non-blocking; drop oldest if full)
        self._ensure_worker(stream_id)
        q = self._sct_queues[stream_id]
        try:
            q.put_nowait(sae_msg)
        except queue.Full:
            logger.warning(f"SCT queue full; dropping oldest frame for {stream_id}")
            # keep latency bounded: drop oldest
            try:
                _ = q.get_nowait()
                q.task_done()
            except queue.Empty:
                pass
            try:
                q.put_nowait(sae_msg)
            except queue.Full:
                logger.warning(f"SCT queue still full; dropping frame for {stream_id}")


        # Save the matched results, prune the tracklets in MCTTrackbase
            
        # inference_time_us = (time.monotonic_ns() - inference_start) // 1000

        # return self._create_output(stream_id,inference_time_us,sae_msg)
        
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