import numpy as np
import logging
from visionapi_yq.messages_pb2 import SaeMessage,TrackletsByCamera,Tracklet,Trajectory
from visionlib.pipeline.tools import get_raw_frame_data

class Trackletdatabase:
    '''Keeps Trajectory() messages'''
    def __init__(self,logger):
        self.data = Trajectory()
        self.logger = logger  
        self.remove_dict = {}
        self.matched_dict = {}

    def append(self,input_img, stream_id, sae_msg:SaeMessage):
        '''Update the tracklet database based on sae_msg'''

        if not sae_msg or not sae_msg.trajectory:
            raise ValueError("Invalid sae_msg: Missing trajectory")      

        height = input_img.shape[0]
        width = input_img.shape[1]
        num_new_tracklet = 0
        num_exist_tracklet = 0
        if stream_id not in self.data.cameras:
            self.data.cameras[stream_id].CopyFrom(TrackletsByCamera())
        if stream_id not in self.remove_dict:
            self.remove_dict[stream_id] = []

        for idx, track_id in enumerate(sae_msg.trajectory.cameras[stream_id].tracklets):
            current_tracklet = sae_msg.trajectory.cameras[stream_id].tracklets[track_id]
            if track_id not in self.remove_dict[stream_id]:
                if (track_id not in self.data.cameras[stream_id].tracklets):
                    self.data.cameras[stream_id].tracklets[track_id].CopyFrom(current_tracklet)
                    self.data.cameras[stream_id].tracklets[track_id].start_time = sae_msg.frame.timestamp_utc_ms

                    self.data.cameras[stream_id].tracklets[track_id].end_time = sae_msg.frame.timestamp_utc_ms
                    num_new_tracklet+=1
                else:
                    self.data.cameras[stream_id].tracklets[track_id].end_time = sae_msg.frame.timestamp_utc_ms
                    previous_feature = np.array(self.data.cameras[stream_id].tracklets[track_id].mean_feature)
                    out_feature = np.array(current_tracklet.mean_feature) # current tracklet should only contain the feature in current timestamp 
                    number_det = len(self.data.cameras[stream_id].tracklets[track_id].detections_info)

                    if number_det == 0:
                        updated_mean_feature = out_feature
                    else:
                        updated_mean_feature = (number_det - 1) / number_det * previous_feature + (1 / number_det) * out_feature

                    # Reassign the updated mean_feature to the repeated field
                    del self.data.cameras[stream_id].tracklets[track_id].mean_feature[:]
                    self.data.cameras[stream_id].tracklets[track_id].mean_feature.extend(updated_mean_feature)

                    # General information update
                    detection = self.data.cameras[stream_id].tracklets[track_id].detections_info.add()
                    for detection_info in current_tracklet.detections_info:
                        detection = self.data.cameras[stream_id].tracklets[track_id].detections_info.add()
                        detection.CopyFrom(detection_info)

                    self.data.cameras[stream_id].tracklets[track_id].status = 'Active'
                    num_exist_tracklet+=1
        self.logger.debug(f'num of new tracklet {num_new_tracklet}, num of exist tracklet {num_exist_tracklet}')

    def tracklet_status_update(self,stream_id,sae_msg:SaeMessage):
        for idx, track_id in enumerate(self.data.cameras[stream_id].tracklets):
            gap_time = max(sae_msg.frame.timestamp_utc_ms - self.data.cameras[stream_id].tracklets[track_id].end_time,0)
            if gap_time > 500:
                self.data.cameras[stream_id].tracklets[track_id].status = 'Searching'
            if gap_time > 10000:
                self.data.cameras[stream_id].tracklets[track_id].status = 'Lost'

    def prune(self,stream_id):

        if stream_id not in self.data.cameras:
            self.logger.warning(f"No tracklets found for stream_id {stream_id}")
            return 
    
        # Collect tracklet IDs to remove
        current_remove_ids = [
        track_id for track_id in self.data.cameras[stream_id].tracklets
        if (self.data.cameras[stream_id].tracklets[track_id].status == 'Lost') or 
           (self.data.cameras[stream_id].tracklets[track_id].age > 50)
        ]

        if stream_id not in self.remove_dict:
            self.remove_dict[stream_id] = []
        
        new_remove_ids = list(set(current_remove_ids) - set(self.remove_dict[stream_id]))

        self.remove_dict[stream_id].extend(new_remove_ids)
            # Remove the collected tracklets

        for track_id in new_remove_ids:
            del self.data.cameras[stream_id].tracklets[track_id]
            self.logger.info(f"Pruned tracklet {track_id} from stream_id {stream_id}")

    def matched_dict_initalize(self,config):
        for stream_id in config.input_stream_ids:
            self.matched_dict[stream_id] = {}

    def matching_result_process(self,reid_dict):
        '''To add the reid_dict result into self.matched_dict'''
        for cam_id in reid_dict:
            if cam_id == 'c001':
                stream_key = 'stream1'
            elif cam_id == 'c002':
                stream_key = 'stream2'
            else:
                raise ValueError('Cam id not avilable')
                
            for track_id in reid_dict[cam_id]:
                if track_id not in self.matched_dict[stream_key]:
                    self.matched_dict[stream_key][track_id] = {}
                
                self.matched_dict[stream_key][track_id]['ori_track_id'] = track_id
                self.matched_dict[stream_key][track_id]['dis'] = reid_dict[cam_id][track_id]['dis']
                self.matched_dict[stream_key][track_id]['new_track_id'] = reid_dict[cam_id][track_id]['id']