import numpy as np
from numba import njit
import logging
from typing import Dict
from visionlib.pipeline.tools import get_raw_frame_data
from .matching_tool import CostMatrix,calc_reid
from .trackletdatabase import Trackletdatabase
import sys
sys.path.append('../')
from visionapi_yq.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet

def tracklet_status_update(stream_id,sae_msg: SaeMessage):
    time_now = sae_msg.frame.timestamp_utc_ms
    for track_id in sae_msg.trajectory.cameras[stream_id].tracklets:
        if (sae_msg.trajectory.cameras[stream_id].tracklets[track_id].age > 30) and (time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time < 100):
            #60000 is just a assuming time in here, this value is get from camera link model
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].status = 'Inactive' # Lost1 means temporal lost
        elif (time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time > 200000) and (time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time < 5000000):
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].status = 'Searching'
        elif time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time > 500000:
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].status = 'Lost'

    return sae_msg


def tracklet_match(config, logger,image,tracklets1:SaeMessage,tracklets2:SaeMessage):

    filtered_tracklets1 = tracklet_filter(logger,image, tracklets1)
    filtered_tracklets2 = tracklet_filter(logger, image, tracklets2)
    logger.info(f'tracklets1 length before filter {len(tracklets1)}, and tracklets1 length after filter {len(filtered_tracklets1)}')
    logger.info(f'tracklets2 length before filter {len(tracklets2)}, and tracklets2 length after filter {len(filtered_tracklets2)}')


    if len(filtered_tracklets1) != 0 and len(filtered_tracklets2) != 0:

        cm = CostMatrix(filtered_tracklets1,filtered_tracklets2)
        dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_statuses, g_statuses, q_class_ids, g_class_ids = cm.cost_matrix(metric = config.matching_metric)
        print(dismat)

        reid_dict = {}
        if dismat.size > 0:
            reid_dict,rm_dict = calc_reid(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, q_statuses, g_statuses, g_times, q_class_ids, g_class_ids,dis_thre=config.dis_thre,dis_remove=config.dis_remove,dis_alpha=config.dis_alpha,dis_beta=config.dis_beta)

        if reid_dict != {}:
            print(reid_dict)
            with open("reid_dict.txt", "w") as txt_file:
                txt_file.write(str(reid_dict))
    
        return reid_dict



@njit
def calculate_distance(start_bbox, last_bbox):
    # Compute the central points of the bounding boxes
    start_central_x = (start_bbox[0] + start_bbox[2]) / 2
    start_central_y = (start_bbox[1] + start_bbox[3]) / 2

    last_central_x = (last_bbox[0] + last_bbox[2]) / 2
    last_central_y = (last_bbox[1] + last_bbox[3]) / 2

    # Calculate the Euclidean distance
    distance = np.sqrt(
        (last_central_x - start_central_x) ** 2 +
        (last_central_y - start_central_y) ** 2
    )
    return distance


def tracklet_filter(logger, img, tracklets: Dict[str, Tracklet]):
    """
    Filters tracklets based on their duration and movement criteria.

    Args:
        img: The input image used to calculate movement in pixels (for height reference).
        tracklets: A dictionary of Tracklet objects keyed by their IDs.

    Returns:
        A filtered dictionary of tracklets meeting the duration and movement criteria.
    """
    filtered_tracklets = {}

    for track_id, tracklet in tracklets.items():
        # Calculate the duration of the tracklet
        time_duration_ms = tracklet.end_time - tracklet.start_time


        # Filter out tracklets with a duration less than 0.5 seconds (500 ms)
        if time_duration_ms < 500:
            logger.debug('time_duration_ms < 500')
            logger.debug(time_duration_ms)
            continue

        # Ensure there are enough detections to calculate movement
        if len(tracklet.detections_info) < 2:
            logger.debug('detection info not complete')
            continue

        # Extract bounding box details
        start_bbox = np.array([
            tracklet.detections_info[0].bounding_box.min_x,
            tracklet.detections_info[0].bounding_box.min_y,
            tracklet.detections_info[0].bounding_box.max_x,
            tracklet.detections_info[0].bounding_box.max_y,
        ])
        last_bbox = np.array([
            tracklet.detections_info[-1].bounding_box.min_x,
            tracklet.detections_info[-1].bounding_box.min_y,
            tracklet.detections_info[-1].bounding_box.max_x,
            tracklet.detections_info[-1].bounding_box.max_y,
        ])

        # Use the Numba-accelerated function
        distance = calculate_distance(start_bbox, last_bbox)
        
        # Filter out tracklets that move less than 1/5th of the image height
        if distance < (1 / 8):
            logger.debug('car is stastic')
            tracklet.age += 1
            logger.debug(distance)
            continue

        # If the tracklet meets all criteria, add it to the filtered dictionary
        filtered_tracklets[track_id] = tracklet

    
    return filtered_tracklets



def unpack_proto(sae_message_bytes):
    sae_msg = SaeMessage()
    sae_msg.ParseFromString(sae_message_bytes)

    input_frame = sae_msg.frame
    input_image = get_raw_frame_data(input_frame)

    return input_image, sae_msg

