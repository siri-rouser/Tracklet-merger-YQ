import numpy as np

from visionlib.pipeline.tools import get_raw_frame_data
from .matching_tool import CostMatrix,calc_reid
from .trackletdatabase import Trackletdatabase
import sys
sys.path.append('../')
from visionapi_yq.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet

def tracklet_info_update(stream_id,tracking_output_array,out_features,out_age,sae_msg: SaeMessage):
    if stream_id not in sae_msg.trajectory.cameras:
        sae_msg.trajectory.cameras[stream_id].CopyFrom(TrackletsByCamera())

    # tracking_output_arrary = [x1, y1, x2, y2, track_id, confidence,class_id]
    tracklet = Tracklet()

    for index,output_array in enumerate(tracking_output_array):
        x1, y1, x2, y2, track_id, confidence, class_id = output_array
        feature = out_features[index]
        track_id = str(track_id)
        if track_id not in sae_msg.trajectory.cameras[stream_id].tracklets:
            # Create a new Tracklet if it doesn't exist
            tracklet = Tracklet()
            tracklet.mean_feature.extend(out_features[index])
            tracklet.status = 'Active'
            tracklet.start_time = sae_msg.frame.timestamp_utc_ms
            tracklet.end_time = sae_msg.frame.timestamp_utc_ms
            tracklet.age = out_age[index]
            # for detection_info
            detection = tracklet.detections_info.add()
            detection.bounding_box.min_x = x1
            detection.bounding_box.min_y = y1
            detection.bounding_box.max_x = x2

            detection.bounding_box.max_y = y2
            detection.confidence = confidence
            detection.class_id = int(class_id)
            detection.feature.extend(out_features[index])

            # Add the new tracklet to the tracklets map
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].CopyFrom(tracklet)
        else:
            # Update existing Tracklet
            existing_tracklet = sae_msg.trajectory.cameras[stream_id].tracklets[track_id]
            previous_feature = np.array(existing_tracklet.mean_feature)
            out_feature = np.array(feature)

            number_det = len(existing_tracklet.detections_info)
            updated_mean_feature = (number_det - 1) / number_det * previous_feature + (1 / number_det) * out_feature
            
            # Reassign the updated mean_feature to the repeated field
            del existing_tracklet.mean_feature[:]
            existing_tracklet.mean_feature.extend(updated_mean_feature)
            
            # for detection_info
            detection = sae_msg.trajectory.cameras[stream_id].tracklets[track_id].detections_info.add()
            detection.bounding_box.min_x = x1
            detection.bounding_box.min_y = y1
            detection.bounding_box.max_x = x2
            detection.bounding_box.max_y = y2
            detection.confidence = confidence
            detection.class_id = int(class_id)
            detection.feature.extend(out_features[index])
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].age = out_age[index]
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time = sae_msg.frame.timestamp_utc_ms
        
        # tracklet[track_id].detections_info.bounding_box.min_x = x1
    return sae_msg


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


def tracklet_match(tracklets1:SaeMessage,tracklets2:SaeMessage):
    print("Number of tracklets in stream1:", len(tracklets1))
    print("Number of tracklets in stream2:", len(tracklets2))

    if len(tracklets1) != 0 and len(tracklets2) != 0:

        cm = CostMatrix(tracklets1,tracklets2)
        dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_statuses, g_statuses = cm.cost_matrix(metric = 'Cosine_Distance')
        print(dismat)

        # NOTE: inspect values, the dismat look not good?
        # Should we only search for tracklets in searching and trackleyts in active? 
        # change the data source to get better results
        reid_dict = {}
        if dismat.size > 0:
            reid_dict,rm_dict = calc_reid(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, q_statuses, g_statuses, g_times)
        print(reid_dict)



def unpack_proto(sae_message_bytes):
    sae_msg = SaeMessage()
    sae_msg.ParseFromString(sae_message_bytes)

    input_frame = sae_msg.frame
    input_image = get_raw_frame_data(input_frame)

    return input_image, sae_msg

