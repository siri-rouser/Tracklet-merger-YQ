import pickle
import logging
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
import sys
from typing import Dict, Tuple, List

sys.path.append('../')
from visionapi_yq.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet,ZoneStatus
# from sklearn.neighbors import KernelDensity

class CostMatrix:

    def __init__(self,tracklets1:Dict[str,Tracklet],tracklets2:Dict[str,Tracklet],cam1,cam2,logger:logging):
        self.query_tracklet = tracklets1
        self.gallery_tracklet = tracklets2
        self.cam1 = cam1
        self.cam2 = cam2
        self.logger = logger

    def cost_matrix(self,metric):
        q_feats, q_track_ids, q_cam_ids, q_times, q_entry_zn, q_exit_zn = self._extract_tracklet_data(self.query_tracklet, cam_id = self.cam1)
        g_feats, g_track_ids, g_cam_ids, g_times, g_entry_zn, g_exit_zn = self._extract_tracklet_data(self.gallery_tracklet, cam_id = self.cam2)
        
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

        # zone is a int variable
        return distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zn, q_exit_zn, g_entry_zn, g_exit_zn 
    
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
    

    def _extract_tracklet_data(self, tracklets:Dict[str,Tracklet],cam_id):
        feats, track_ids, cam_ids, times, entry_zn, exit_zn = [], [], [], [], [], []

        with torch.no_grad():
            for track_id, tracklet in tracklets.items():
                feat_tensor = torch.tensor(tracklet.mean_feature, dtype=torch.float).unsqueeze(0)
                feats.append(feat_tensor)
                track_ids.append(int(float(track_id)))
                cam_ids.append(int(cam_id[-1])) # stream1, stream2, etc.
                times.append([tracklet.start_frame, tracklet.end_frame])
                entry_zn.append([tracklet.zone.entry_zone_id, tracklet.zone.entry_zone_type])
                exit_zn.append([tracklet.zone.exit_zone_id, tracklet.zone.exit_zone_type])

            feats = torch.cat(feats, 0) if feats else torch.empty((0, 0))
            track_ids = np.asarray(track_ids,dtype=np.int32)
            cam_ids = np.asarray(cam_ids)

        self.logger.debug(f'Got features for set, obtained {feats.size(0)}-by-{feats.size(1)} matrix' if feats.size(0) > 0 else 'No features found.')

        return feats, track_ids, cam_ids, times, entry_zn, exit_zn




class ReIDCalculator:
    def __init__(self, logger: logging, dis_thre=0.7, dis_remove=0.9, dis_alpha=1.0, dis_beta=1.0, kde_threshold = 5e-4, clm:Dict[str, Dict[str, KernelDensity]] = None):
        self.logger = logger
        self.clm = clm
        self.kde_threshold = kde_threshold
        self.dis_thre = dis_thre
        self.dis_remove = dis_remove
        self.dis_alpha = dis_alpha
        self.dis_beta = dis_beta

    def _kde_filter(self, dismat, index, orders, q_time, g_times,q_cam_id, g_cam_ids):
        directions = []
        for order in orders:
            if (q_time[0] - g_times[order][1]) > 0: # g --> q g_exit and q_entry
                trans_time = (q_time[0] - g_times[order][1]) / 15 # 15 is the fps
                kde = self.clm[f"{q_cam_id}_to_{g_cam_ids[0]}"]['kde']
                directions.append(f'{q_cam_id}_to_{g_cam_ids[0]}')
            else: # q --> g q_exit and g_entry
                trans_time = (g_times[order][0] - q_time[1]) / 15
                kde = self.clm[f"{g_cam_ids[0]}_to_{q_cam_id}"]['kde']
                directions.append(f'{g_cam_ids[0]}_to_{q_cam_id}')
            pdf = np.exp(kde.score_samples(np.array(trans_time).reshape(-1, 1)))

            if pdf < self.kde_threshold:
                dismat[index][order] += 1
            else:
                dismat[index][order] = self.dis_alpha * dismat[index][order] - self.dis_beta * pdf
        return dismat, directions
    
    def _zone_remove(self, q_entry_zn, q_exit_zn, g_entry_zn, g_exit_zn, q_cam_id, g_cam_ids, order,directions):
        '''
        q_entry_zn: [entry_zone_id, entry_zone_type] for query tracklet
        q_exit_zn: [exit_zone_id, exit_zone_type] for query tracklet
        '''
        zone_remove = []
        for idx,ord in enumerate(order):
            direction:List[str] = directions[idx]
            entry_cam = direction.split('_')[0]
            exit_cam = direction.split('_')[2]  
            entry_zone_gt, exit_zone_gt = (self.clm[direction]['entry_exit_pair'][0],self.clm[direction]['entry_exit_pair'][1]) # e.g. {'1_to_2': {'entry_exit_pair': {'1': '1', '2': '2'}, 'time_pair': [[0, 100]]}}
            
            if entry_cam == q_cam_id and exit_cam == g_cam_ids[ord]:
                # q_entry_zn is the entry zone for query tracklet, g_entry_zn is the entry zone for gallery tracklet
                if (q_entry_zn[0] != entry_zone_gt and q_entry_zn[1] == ZoneStatus.ENTRY) or (g_exit_zn[ord][0] != exit_zone_gt and g_exit_zn[ord][1] == ZoneStatus.EXIT):
                    zone_remove.append(True)
                else:
                    zone_remove.append(False)
            elif entry_cam == g_cam_ids[ord] and exit_cam == q_cam_id:
                if (g_entry_zn[ord][0] != entry_zone_gt and g_entry_zn[ord][1] == ZoneStatus.ENTRY) or (q_exit_zn[0] != exit_zone_gt and q_exit_zn[1] == ZoneStatus.EXIT):
                    zone_remove.append(True)
                else:
                    zone_remove.append(False)
            else:
                self.logger.warning(f"Unexpected camera direction: {direction} for tracklet {q_cam_id} to {g_cam_ids[ord]}")
                zone_remove.append(False)
                
        return np.array(zone_remove)

    def calc(self, dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zn, q_exit_zn, g_entry_zn, g_exit_zn):
        '''
        dismat: distance matrix
        q_track_ids: query tracklet ids
        q_cam_ids: query camera ids e.g. int(1), 2, 3
        q_times: [start_frame, end_frame] for query tracklet
        q_entry_zn: [entry_zone_id, entry_zone_type] for query tracklet
        q_exit_zn: [exit_zone_id, exit_zone_type] for query tracklet
        '''
        rm_dict = {}
        reid_dict = {}
        indices = np.argsort(dismat, axis=1)

        for index, q_track_id in enumerate(q_track_ids):
            q_cam_id = q_cam_ids[index]
            q_time = q_times[index]
            order = indices[index]

            if self.clm is None:
                # directly solve the cost-matrix
                self.logger.warning(f"Camera link model not found")
                remove = (g_cam_ids[order] == q_cam_id) | (dismat[index][order] > self.dis_thre) 
            else:
                directions,dismat = self._kde_filter(dismat, index, order, q_time, g_times,q_cam_id, g_cam_ids)
                zone_remove = self._zone_remove(q_entry_zn[index], q_exit_zn[index], g_entry_zn, g_exit_zn,q_cam_id, g_cam_ids,order,directions)

                remove = (g_cam_ids[order] == q_cam_id) | \
                        (dismat[index][order] > self.dis_thre) | \
                        zone_remove

            keep = np.invert(remove)
            remove_hard = (g_cam_ids[order] == q_cam_id) | \
                          (dismat[index][order] > self.dis_remove)
            keep_hard = np.invert(remove_hard)

            if not np.any(keep_hard):
                rm_dict.setdefault(q_cam_id, {})[q_track_id] = True
                continue

            sel_g_dis = dismat[index][order][keep]
            sel_g_track_ids = g_track_ids[order][keep]
            sel_g_cam_ids = g_cam_ids[order][keep]

            if sel_g_dis.size > 0:
                reid_dict.setdefault(q_cam_id, {}).setdefault(q_track_id, {"dis": float('inf'), "id": -1})
                if reid_dict[q_cam_id][q_track_id]["dis"] > sel_g_dis.min():
                    reid_dict[q_cam_id][q_track_id] = {"dis": sel_g_dis.min(), "id": q_track_id} # reid_dict[cam1][q_track_id] = {"dis": dis, "id": q_track_id}

            for i, (sel_id, sel_cam, sel_dis) in enumerate(zip(sel_g_track_ids, sel_g_cam_ids, sel_g_dis)):
                reid_dict.setdefault(sel_cam, {}).setdefault(sel_id, {"dis": float('inf'), "id": -1})
                if reid_dict[sel_cam][sel_id]["dis"] > sel_dis:
                    reid_dict[sel_cam][sel_id] = {"dis": sel_dis, "id": q_track_id} # once matched: reid_dict[cam2][sel_id] = {"dis": dis, "id": q_track_id}

        return reid_dict, rm_dict