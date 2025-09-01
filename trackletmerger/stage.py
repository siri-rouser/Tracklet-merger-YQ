import logging
import signal
import threading
import time
from typing import List, Tuple
import json

from prometheus_client import Counter, Histogram, start_http_server
from visionlib.pipeline.consumer import RedisConsumer
from visionlib.pipeline.publisher import RedisPublisher

from .config import TrackletMergerConfig
from .trackletmerger import TrackletMerger
import atexit
from visionapi_yq.messages_pb2 import Detection, SaeMessage

logger = logging.getLogger(__name__)

REDIS_PUBLISH_DURATION = Histogram('my_stage_redis_publish_duration', 'The time it takes to push a message onto the Redis stream',
                                   buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))

# def filter_match_dict(match_dict):
#     '''
#     Remove duplicates based on 'new_track_id' within each 'stream_key'.
#     Keeps the entry with the lowest 'dis' and logs any removals.
#     '''

#     for stream_key, tracks in match_dict.items():
#         new_track_id_map = {}

#         # Find lowest dis for each new_track_id within this stream_key
#         for track_id, data in tracks.items():
#             new_id = data['new_track_id']
#             dis = data['dis']

#             if new_id not in new_track_id_map:
#                 new_track_id_map[new_id] = (track_id, dis)
#             else:
#                 existing_track_id, existing_dis = new_track_id_map[new_id]
#                 if dis < existing_dis:
#                     # Current entry has lower dis; mark previous one for deletion
#                     logger.info(
#                         f"[{stream_key}] Removing duplicate tracklet {existing_track_id} "
#                         f"(dis={existing_dis:.4f}) due to higher dis compared to {track_id} (dis={dis:.4f})"
#                     )
#                     match_dict[stream_key][existing_track_id]['to_remove'] = True
#                     new_track_id_map[new_id] = (track_id, dis)
#                 else:
#                     # Current entry has higher dis; mark current entry for deletion
#                     logger.info(
#                         f"[{stream_key}] Removing duplicate tracklet {track_id} "
#                         f"(dis={dis:.4f}) due to higher dis compared to {existing_track_id} (dis={existing_dis:.4f})"
#                     )
#                     match_dict[stream_key][track_id]['to_remove'] = True

#     # Remove marked entries
#     for stream_key, tracks in match_dict.items():
#         remove_keys = [track_id for track_id, data in tracks.items() if data.get('to_remove')]
#         for track_id in remove_keys:
#             del match_dict[stream_key][track_id]

#     return match_dict


# def print_match_dict_without_detections(match_dict):
#     json_path = 'match_dict.json'
#     match_dict_no_detections = {}
#     for stream_id, tracks in match_dict.items():
#         print(f"Stream: {stream_id}")
#         match_dict_no_detections[stream_id] = {}
#         for track_id, track_data in tracks.items():
#             # Create a shallow copy excluding 'detections_info'
#             clean_data = {'ori_track_id': int(track_data['ori_track_id']),
#                           'dis': float(track_data['dis']),
#                         'new_track_id': int(track_data['new_track_id'])}
#             match_dict_no_detections[stream_id][int(track_id)] = clean_data
#             print(f"Track ID {track_id}: {clean_data}")

#     with open(json_path, 'w') as json_file:
#         json.dump(match_dict_no_detections, json_file, indent=4)

# def save_results(match_dict,img_size,first_frame_timestamp,save_path):
#     # the result is a matched dict[]
#     # {cam} {id_index} {frame_num} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {xworld} {yworld}
#     # NB is '1' SB is '2'
#     frame_id_list = [] 
#     w = img_size[0]
#     h = img_size[1]
#     time_max = first_frame_timestamp
#     print_match_dict_without_detections(match_dict)

#     match_dict = filter_match_dict(match_dict)

#     with open(save_path, "w") as f:
#         for stream_id in match_dict:
#             cam = 1 if stream_id == 'stream1' else 2
#             for track_id in match_dict[stream_id]:
#                 id_index = match_dict[stream_id][track_id]['new_track_id']
#                 for detection_bbox, timestamp,frame_id in match_dict[stream_id][track_id]['detections_info']:
#                     if timestamp < first_frame_timestamp:
#                         logger.warning(f"Detection timestamp {timestamp} is before the first frame timestamp {first_frame_timestamp}")
#                         continue
#                     if frame_id not in frame_id_list:
#                         frame_id_list.append(frame_id)
#                     time_max = max(timestamp, time_max)
#                     x1 = detection_bbox.min_x * w
#                     y1 = detection_bbox.min_y * h
#                     width = (detection_bbox.max_x - detection_bbox.min_x) * w
#                     height = (detection_bbox.max_y - detection_bbox.min_y) * h
#                     if x1 == 0 and y1 == 0 and width == 0 and height == 0:
#                         logger.warning(f"Detection bbox is zero: {detection_bbox}")
#                         continue
#                     line = f"{cam} {id_index} {frame_id} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {-1} {-1} {track_id}"
#                     f.write(line + "\n")
                    
#     try:
#         completion_ratio = len(frame_id_list) / max(frame_id_list)
#         logger.critical(f'The frame completion ratio: {completion_ratio:.3f}')
#     except:
#         pass

def run_stage():

    stop_event = threading.Event()

    # Register signal handlers
    def sig_handler(signum, _):
        signame = signal.Signals(signum).name
        print(f'Caught signal {signame} ({signum}). Exiting...')
        stop_event.set()

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    # Load config from settings.yaml / env vars
    CONFIG = TrackletMergerConfig()

    logger.setLevel(CONFIG.log_level.value)

    logger.info(f'Starting prometheus metrics endpoint on port {CONFIG.prometheus_port}')

    start_http_server(CONFIG.prometheus_port)

    logger.info(f'Starting geo mapper stage. Config: {CONFIG.model_dump_json(indent=2)}')

    trackletmerger = TrackletMerger(CONFIG,CONFIG.log_level)

    # def exit_handler():
    #     logger.info("Program exiting. Saving results to file...")
    #     match_dict, img_size, first_frame_timestamp = trackletmerger.get_results()
    #     save_results(match_dict, img_size, first_frame_timestamp,CONFIG.save_path)

    # # Save all results to file on exit
    # atexit.register(exit_handler)

    consume = RedisConsumer(CONFIG.redis.host, CONFIG.redis.port, 
                            stream_keys=[f'{CONFIG.redis.input_stream_prefix}:{stream_id}' for stream_id in CONFIG.merging_config.input_stream_ids],
                            block=500)
    # publish = RedisPublisher(CONFIG.redis.host, CONFIG.redis.port)
    
    with consume:
        for stream_key, proto_data in consume():
            if stop_event.is_set():
                break

            if proto_data is None or stream_key is None:
                continue

            if stream_key is not None:
               stream_id = stream_key.split(':')[1]

            trackletmerger.get(stream_id,proto_data)

            # with REDIS_PUBLISH_DURATION.time():
            #     publish(f'{CONFIG.redis.output_stream_prefix}:{stream_id}', output_proto_data)