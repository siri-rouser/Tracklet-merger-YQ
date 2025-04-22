import pickle
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
import sys

sys.path.append('../')
from visionapi_yq.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet
# from sklearn.neighbors import KernelDensity

class CostMatrix:

    def __init__(self,tracklets1:SaeMessage,tracklets2:SaeMessage,logger):
        self.query_tracklet = tracklets1
        self.gallery_tracklet = tracklets2
        self.logger = logger

    def cost_matrix(self,metric):
        q_feats, q_track_ids, q_cam_ids, q_times, q_track_status, q_class_ids, q_entry_zn, q_exit_zn = self._extract_tracklet_data(self.query_tracklet, cam_id = 'c001')
        g_feats, g_track_ids, g_cam_ids, g_times, g_track_status, g_class_ids, g_enrty_zn, g_exit_zn = self._extract_tracklet_data(self.gallery_tracklet, cam_id = 'c002')
        
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
        return distmat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_track_status, g_track_status, q_class_ids, g_class_ids, q_entry_zn, q_exit_zn, g_enrty_zn, g_exit_zn 
    
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
        feats, track_ids, cam_ids, track_status, times, class_ids, entry_zn, exit_zn = [], [], [], [], [], [], [], []

        with torch.no_grad():
            for track_id in tracklets.keys():
                feat_tensor = torch.tensor(tracklets[track_id].mean_feature, dtype=torch.float).unsqueeze(0)
                feats.append(feat_tensor)
                track_ids.append(int(float(track_id)))
                cam_ids.append(cam_id)
                times.append([tracklets[track_id].start_time, tracklets[track_id].end_time])
                track_status.append(tracklets[track_id].status)
                class_ids.append(tracklets[track_id].detections_info[0].object_id)
                entry_zn.append(tracklets[track_id].entry_zone)
                exit_zn.append(tracklets[track_id].exit_zone)

            feats = torch.cat(feats, 0) if feats else torch.empty((0, 0))
            track_ids = np.asarray(track_ids,dtype=np.int32)
            cam_ids = np.asarray(cam_ids)

        self.logger.debug(f'Got features for set, obtained {feats.size(0)}-by-{feats.size(1)} matrix' if feats.size(0) > 0 else 'No features found.')

        return feats, track_ids, cam_ids, times, track_status, class_ids, entry_zn, exit_zn




class ReIDCalculator:
    def __init__(self, dis_thre=0.7, dis_remove=0.9, dis_alpha=1.0, dis_beta=1.0):
        self.dis_thre = dis_thre
        self.dis_remove = dis_remove
        self.dis_alpha = dis_alpha
        self.dis_beta = dis_beta
        data = np.array([0.5, 1, 0.9, 0.1, 0.5, 0.24, 0.4, 1.2, 0.1]).reshape(-1, 1)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(data)

    def _status_remove_gen(self, q_status, g_statuses, order):
        type_remove = []
        if q_status == 'Lost':
            type_remove = [True for _ in order]
        elif q_status == 'Active':
            type_remove = [g_statuses[ord] != 'Searching' for ord in order]
        else:  # Searching
            type_remove = [g_statuses[ord] != 'Active' for ord in order]
        return np.array(type_remove)

    def _kde_filter(self, dismat, index, orders, q_time, g_times):
        for order in orders:
            if (q_time[0] - g_times[order][1]) > 0: 
                trans_time = (q_time[0] - g_times[order][1]) / 1000
            else:
                trans_time = (g_times[order][0] - q_time[1]) / 1000
            pdf = np.exp(self.kde.score_samples(np.array(trans_time).reshape(-1, 1)))

            if pdf < 2e-3:
                dismat[index][order] += 1
            else:
                dismat[index][order] = self.dis_alpha * dismat[index][order] - self.dis_beta * pdf
        return dismat
    
    def _zone_remove(self, q_entry_zn, q_exit_zn, g_entry_zn, g_exit_zn,order):
        zone_remove = []
        for ord in order:
            # for those two example cameras: the zone 01 is paired with zone 01 on the other camera
            if (q_entry_zn == '1' and g_exit_zn[ord] == '1') or \
            (q_exit_zn == '1' and g_entry_zn[ord] == '1'):
                zone_remove.append(False)
            else:
                zone_remove.append(True)
        return np.array(zone_remove)

    def calc(self, dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, q_statuses, g_statuses, g_times, q_class_ids, g_class_ids, q_entry_zn, q_exit_zn, g_enrty_zn, g_exit_zn):
        # q_tracks are from stream1 and g_tracks are from stream2
        rm_dict = {}
        reid_dict = {}
        indices = np.argsort(dismat, axis=1)

        for index, q_track_id in enumerate(q_track_ids):
            q_cam_id = q_cam_ids[index]
            q_class_id = q_class_ids[index]
            q_time = q_times[index]
            q_status = q_statuses[index]
            order = indices[index]

            status_remove = self._status_remove_gen(q_status, g_statuses, order)
            dismat = self._kde_filter(dismat, index, order, q_time, g_times)
            zone_remove = self._zone_remove(q_entry_zn[index], q_exit_zn[index], g_enrty_zn, g_exit_zn,order)

            remove = (g_cam_ids[order] == q_cam_id) | \
                     (dismat[index][order] > self.dis_thre) | \
                     status_remove | \
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