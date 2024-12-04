import numpy as np
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet

class Trackletdatabase:
    def __init__(self):
        self.data = SaeMessage() 
    #NOTE:Current I am thinking about append the sae_message from each saeframe as the database, so in that case i donot need the buffer. will that works?

    def append(self,input_img, stream_id, sae_msg:SaeMessage):

        if not sae_msg or not sae_msg.trajectory:
            raise ValueError("Invalid sae_msg: Missing trajectory")

        height = input_img.shape[0]
        width = input_img.shape[1]
        if stream_id not in self.data.trajectory.cameras:
            self.data.trajectory.cameras[stream_id].CopyFrom(TrackletsByCamera())

        for idx, track_id in enumerate(sae_msg.trajectory.cameras[stream_id].tracklets):
            current_tracklet = sae_msg.trajectory.cameras[stream_id].tracklets[track_id]

            if self._tracklet_filter(current_tracklet):

                if track_id not in self.data.trajectory.cameras[stream_id].tracklets:
                    self.data.trajectory.cameras[stream_id].tracklets[track_id].CopyFrom(current_tracklet)
                else:
                    # database_tracklet = Tracklet()
                    database_tracklet = self.data.trajectory.cameras[stream_id].tracklets[track_id]
                    previous_feature = np.array(database_tracklet.mean_feature)
                    out_feature = np.array(current_tracklet.mean_feature) # current tracklet should only contain the feature in current timestamp 

                    number_det = len(database_tracklet.detections_info)
                    if number_det == 0:
                        updated_mean_feature = out_feature
                    else:
                        updated_mean_feature = (number_det - 1) / number_det * previous_feature + (1 / number_det) * out_feature

                    # Reassign the updated mean_feature to the repeated field
                    del database_tracklet.mean_feature[:]
                    database_tracklet.mean_feature.extend(updated_mean_feature)

                    # General information update
                    detection = database_tracklet.detections_info.add()
                    for detection_info in current_tracklet.detections_info:
                        detection = database_tracklet.detections_info.add()
                        detection.CopyFrom(detection_info)

                    database_tracklet.age = int(current_tracklet.age)
                    database_tracklet.end_time = sae_msg.frame.timestamp_utc_ms
            else:
                continue

    def _tracklet_filter(self,tracklet:Tracklet):
        if tracklet.end_time -tracklet.start_time < 500: 
            return False
        else:
            tracklet.detections_info.bounding_box


def tracklet_info_update(stream_id,tracking_output_array,out_features,image,sae_msg: SaeMessage):
    '''
    This function serves tracking_output from the tracker at first and saves/update the tracklet information to SaeMessgae.trajectory

    checked, functionality works ok? but the detection info add seems to be redundant
    '''
    height = image.shape[0]
    width = image.shape[1]
    
    if stream_id not in sae_msg.trajectory.cameras:
        sae_msg.trajectory.cameras[stream_id].CopyFrom(TrackletsByCamera())

    # tracking_output_arrary = [x1, y1, x2, y2, track_id, confidence, class_id, age]
    tracklet = Tracklet()

    for index,output_array in enumerate(tracking_output_array):
        x1, y1, x2, y2, track_id, confidence, class_id, age = output_array
        feature = out_features[index]
        track_id = str(track_id)
        if track_id not in sae_msg.trajectory.cameras[stream_id].tracklets:
            tracklet = Tracklet() # Create a new Tracklet if it doesn't exist
            tracklet.mean_feature.extend(out_features[index])
            tracklet.status = 'Active'
            tracklet.start_time = sae_msg.frame.timestamp_utc_ms
            tracklet.end_time = sae_msg.frame.timestamp_utc_ms
            tracklet.age = int(age)

            # for detection_info
            #NOTE: double-check if it is necessary to add the detection information
            detection = tracklet.detections_info.add()
            detection.bounding_box.min_x = float(x1) / width
            detection.bounding_box.min_y = float(y1) / height
            detection.bounding_box.max_x = float(x2) / width
            detection.bounding_box.max_y = float(y2) / height
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
            
            # General information update
            detection = sae_msg.trajectory.cameras[stream_id].tracklets[track_id].detections_info.add()
            detection.bounding_box.min_x = float(x1) / width
            detection.bounding_box.min_y = float(y1) / height
            detection.bounding_box.max_x = float(x2) / width
            detection.bounding_box.max_y = float(y2) / height
            detection.confidence = confidence
            detection.class_id = int(class_id)
            detection.feature.extend(out_features[index])
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].age = int(age)
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time = sae_msg.frame.timestamp_utc_ms
        
    return sae_msg