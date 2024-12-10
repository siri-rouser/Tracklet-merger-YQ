import pickle
import numpy as np
import torch
import sys

sys.path.append('../')
from visionapi_yq.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet
# from sklearn.neighbors import KernelDensity

class CostMatrix:

    def __init__(self,tracklets1:SaeMessage,tracklets2:SaeMessage):
        self.query_tracklet = tracklets1
        self.gallery_tracklet = tracklets2

    def cost_matrix(self,metric):
        q_feats, q_track_ids, q_cam_ids, q_times, q_track_status, q_class_ids = self._extract_tracklet_data(self.query_tracklet, cam_id = 'c001')
        g_feats, g_track_ids, g_cam_ids, g_times, g_track_status, g_class_ids = self._extract_tracklet_data(self.gallery_tracklet, cam_id = 'c002')
        
        if q_feats is None or g_feats is None or q_feats.size(0) == 0 or g_feats.size(0) == 0:
            distmat = []
        else:
            if metric == 'Euclidean_Distance':    
                distmat = self.euclidean_distance(q_feats, g_feats)
            elif metric == 'Cosine_Distance':
                distmat = self.cosine_distance(q_feats,g_feats)
            else:
                sys.exit('Please input the right metric')

        q_times = np.asarray(q_times)
        g_times = np.asarray(g_times)
        g_class_ids = np.array(g_class_ids)
        q_class_ids = np.array(q_class_ids)
        # zone is a int variable
        return distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_track_status, g_track_status, q_class_ids, g_class_ids 
    
    def _track_operation(self,tracklet_path):
        feats, track_ids, cam_ids, entry_zones, exit_zones = [], [], [], [], []
        times = []
        with open(tracklet_path,'rb') as f:
            track_info = pickle.load(f)
        
        with torch.no_grad():
            for track_id,value in track_info.items():
                feat_tensor = torch.tensor(value['feat'], dtype=torch.float).unsqueeze(0)# torch.unsequezze(0) add one dimenson to let it be [1x1024] ratherthan [1024]
                feats.append(feat_tensor)
                track_ids.append(track_id)
                cam_ids.append(int(value['cam'][-3:]))
                times.append([value['start_time'],value['end_time']])
                if 'entry_zone_cls' in list(value.keys()):
                    entry_zones.append([value['entry_zone_cls'], value['entry_zone_id']])
                    exit_zones.append([value['exit_zone_cls'], value['exit_zone_id']])

            feats = torch.cat(feats,0)
            track_ids = np.asarray(track_ids)
            cam_ids = np.asarray(cam_ids)
       
        print('Got features for set, obtained {}-by-{} matrix'.format(feats.size(0), feats.size(1)))

        return feats,track_ids,cam_ids,times,entry_zones,exit_zones
    
    def euclidean_distance(self, q_feats, g_feats):
        m, n = q_feats.size(0), g_feats.size(0)
        distmat = torch.pow(q_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(g_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # torch.power --> torch.sum(dim=1) make it to be one column, those two step calculate the L2 norm--> torch.expand make it expand to m,n(base on the size of qf and gf)
        # torch.t() is the transposed function
        distmat.addmm_(1, -2, q_feats, g_feats.t()) # here calculate the a^2+b^2-2ab 
        # in here, 1 is alpha, -2 is beta: 1*dismat -2*qf*gf.t()
        distmat = distmat.numpy()

        return distmat
    
    def cosine_distance(self,q_feats, g_feats):
        q_feats = torch.nn.functional.normalize(q_feats, p=2, dim=1) # p=2 means sqrt(||q_feats||^2)
        g_feats = torch.nn.functional.normalize(g_feats, p=2, dim=1)

        # Compute the cosine similarity
        cosine_sim = torch.mm(q_feats, g_feats.t())

        # Since cosine distance is 1 - cosine similarity
        distmat = 1 - cosine_sim

        # Convert the distance matrix from torch tensor to numpy array
        distmat = distmat.numpy()

        return distmat
    

    def _extract_tracklet_data(self, tracklets,cam_id):
        feats, track_ids, cam_ids, track_status, times, class_ids = [], [], [], [], [], []

        with torch.no_grad():
            for track_id in tracklets.keys():
                feat_tensor = torch.tensor(tracklets[track_id].mean_feature, dtype=torch.float).unsqueeze(0)
                feats.append(feat_tensor)
                track_ids.append(track_id)
                cam_ids.append(cam_id)
                times.append([tracklets[track_id].start_time, tracklets[track_id].end_time])
                track_status.append(tracklets[track_id].status)
                class_ids.append(tracklets[track_id].detections_info[0].object_id)

            feats = torch.cat(feats, 0) if feats else torch.empty((0, 0))
            track_ids = np.asarray(track_ids)
            cam_ids = np.asarray(cam_ids)

        print(f'Got features for set, obtained {feats.size(0)}-by-{feats.size(1)} matrix' if feats.size(0) > 0 else 'No features found.')

        return feats, track_ids, cam_ids, times, track_status, class_ids


def status_remove_gen(q_status,g_statuses,order):
    type_remove = []
    if q_status == 'Lost':
        for ord in order:
            type_remove.append(True)
    elif q_status == 'Active':
        for ord in order:
            if g_statuses[ord] == 'Searching':
                type_remove.append(False)
            else:
                type_remove.append(True)
    else: # if q_status is searching
        for ord in order:
            if q_status == 'Searching':
                if g_statuses[ord] == 'Active':
                    type_remove.append(False)
                else:
                    type_remove.append(True)
    return type_remove


def calc_reid(dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, q_statuses, g_statuses, g_times,q_class_ids, g_class_ids, dis_thre=0.7,dis_remove=0.8):
    # dis_thre=0.47,dis_remove=0.57
    # For Euclidean Distance (0.29,0.34)
    # new_id = np.max(g_track_ids)
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(dismat, axis=1) 
    # print(indices)
    # num_q, num_g = dismat.shape    
    for index, q_track_id in enumerate(q_track_ids):

        q_cam_id = q_cam_ids[index]
        q_class_id =q_class_ids[index]
        q_time = q_times[index]
        q_status = q_statuses[index]
        # q_entry_pair = []
        # q_exit_pair = []
        # g_times = np.array([time[0] for time in g_times])
        
        # check is that workable or not 
        order = indices[index] # the real order for the first query 

        # q_entry_pair.append(ZONE_PAIR[q_cam_id-41][g_cam_id-41] for g_cam_id in g_cam_ids[order])
        # q_exit_pair.append(ZONE_PAIR[g_cam_id-41][q_cam_id-41] for g_cam_id in g_cam_ids[order])
        # | is or True | False -> True
        status_remove = status_remove_gen(q_status,g_statuses,order)

        remove = (g_cam_ids[order] == q_cam_id) | \
                (g_class_ids[order] != q_class_id)| \
                (dismat[index][order] > dis_thre) | \
                (status_remove)

        # remove all track g_time < q_time + min_time and g_time > q_time + max_time
        keep = np.invert(remove)
        #print(keep)
        # 是在remove query list，所以是很有意义的！
        # part1 = (g_track_ids[order] == q_track_id)
        # if False not in part1: # nothing is been kept,所有的track都匹配不上
        #     print('All track removed in part1')
        # else:
        #     print(part1)
        # part2 = (g_cam_ids[order] == q_cam_id)
        # if False not in part1: # nothing is been kept,所有的track都匹配不上
        #     print('All track removed in part2')
        # else:
        #     print(part2)
        # part3= (dismat[index][order]>dis_remove)
        # if False not in part1: # nothing is been kept,所有的track都匹配不上
        #     print('All track removed in part3')
        # else:
        #     print('This is part3')
        #     print(part3)
        remove_hard = (g_track_ids[order] == q_track_id) | \
                      (g_cam_ids[order] == q_cam_id) | \
                      (dismat[index][order]>dis_remove)
        # print(remove_hard)
        keep_hard = np.invert(remove_hard)
 
        if True not in keep_hard: # nothing is been kept,所有的track都匹配不上
            # print('NO TRACK LEFT FOR PAIRING')
            if q_cam_id not in list(rm_dict.keys()):
                rm_dict[q_cam_id] = {}
            rm_dict[q_cam_id][q_track_id] = True       

        sel_g_dis = dismat[index][order][keep]
        sel_g_track_ids = g_track_ids[order][keep]
        sel_g_cam_ids = g_cam_ids[order][keep]
        sel_g_track_list = []
        sel_g_camids_list = []
        selg_dis_list = []


        for i in range(sel_g_track_ids.shape[0]): # should be the length of sel_g_ids, it should be 1xlen(shape), in the case it only has one dimension so would be shape
            sel_id = sel_g_track_ids[i]
            sel_cam = sel_g_cam_ids[i]
            sel_dis = sel_g_dis[i]

            sel_g_track_list.append(sel_id)
            sel_g_camids_list.append(sel_cam)
            selg_dis_list.append(sel_dis)
       # print(sel_g_track_ids)
    
    #NOTE: i changed this part, all paired tracklets should refer to q_track_id:)
        if len(selg_dis_list) > 0:
            if q_cam_id in list(reid_dict.keys()):
                if q_track_id in list(reid_dict[q_cam_id]): # second time for the matching! it means find this track again!
                    if reid_dict[q_cam_id][q_track_id]["dis"]>min(selg_dis_list):
                        reid_dict[q_cam_id][q_track_id]["dis"] = min(selg_dis_list)
                        reid_dict[q_cam_id][q_track_id]["id"] = q_track_id
                else:
                    reid_dict[q_cam_id][q_track_id] = {"dis":min(selg_dis_list),"id":q_track_id}
            else:
                # that is the initalization part
                reid_dict[q_cam_id] = {}
                reid_dict[q_cam_id][q_track_id] = {"dis":min(selg_dis_list),"id":q_track_id} # assigining a new id top this track!!!!!!
        
        # this is the second loop
        for i in range(len(sel_g_track_list)):
            # print(f'cam_id:{sel_g_camids_list[i]}')
            # print(f'track_id:{sel_g_track_list[i]}')
            if sel_g_camids_list[i] in list(reid_dict.keys()):
                if sel_g_track_list[i] in list(reid_dict[sel_g_camids_list[i]]): 
                    if reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]]["dis"]>selg_dis_list[i]:
                        reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]]["dis"] = selg_dis_list[i]
                        reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]]["id"] = q_track_id
                else:
                       reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]] = {"dis":selg_dis_list[i],"id":q_track_id}
            else:
                reid_dict[sel_g_camids_list[i]] = {}
                reid_dict[sel_g_camids_list[i]][sel_g_track_list[i]] = {"dis":selg_dis_list[i],"id":q_track_id}
        
    return reid_dict,rm_dict