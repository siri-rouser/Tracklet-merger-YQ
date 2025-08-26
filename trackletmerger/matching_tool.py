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
                kde = self.clm[f"{q_cam_id}_to_{g_cam_ids[order]}"]['kde']
                directions.append(f'{q_cam_id}_to_{g_cam_ids[order]}')
            else: # q --> g q_exit and g_entry
                trans_time = (g_times[order][0] - q_time[1]) / 15
                kde = self.clm[f"{g_cam_ids[order]}_to_{q_cam_id}"]['kde']
                directions.append(f'{g_cam_ids[order]}_to_{q_cam_id}')
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
            direction:str = directions[idx]
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
    
    # NOTE: below is the original version(greedy match) of calc function before changing it to Hungarian algorithm
    # def calc(self, dismat,q_track_ids,q_cam_ids, g_track_ids, g_cam_ids, q_times, g_times, q_entry_zn, q_exit_zn, g_entry_zn, g_exit_zn):
    #     '''
    #     dismat: distance matrix
    #     q_track_ids: query tracklet ids
    #     q_cam_ids: query camera ids e.g. int(1), 2, 3
    #     q_times: [start_frame, end_frame] for query tracklet
    #     q_entry_zn: [entry_zone_id, entry_zone_type] for query tracklet
    #     q_exit_zn: [exit_zone_id, exit_zone_type] for query tracklet
    #     '''
    #     # matches = List[(cam_id, track_id, matched_cam_id, matched_track_id, distance)]

    #     rm_dict = {}
    #     reid_dict = {}
    #     matches = []
    #     indices = np.argsort(dismat, axis=1)

    #     for index, q_track_id in enumerate(q_track_ids):
    #         q_cam_id = q_cam_ids[index]
    #         q_time = q_times[index]
    #         order = indices[index]

    #         if self.clm is None:
    #             # directly solve the cost-matrix
    #             self.logger.warning(f"Camera link model not found")
    #             remove = (g_cam_ids[order] == q_cam_id) | (dismat[index][order] > self.dis_thre) 
    #         else:
    #             dismat, directions = self._kde_filter(dismat, index, order, q_time, g_times,q_cam_id, g_cam_ids)
    #             zone_remove = self._zone_remove(q_entry_zn[index], q_exit_zn[index], g_entry_zn, g_exit_zn,q_cam_id, g_cam_ids,order,directions)

    #             remove = (g_cam_ids[order] == q_cam_id) | \
    #                     (dismat[index][order] > self.dis_thre) | \
    #                     zone_remove

    #         keep = np.invert(remove)
    #         remove_hard = (g_cam_ids[order] == q_cam_id) | \
    #                       (dismat[index][order] > self.dis_remove)
    #         keep_hard = np.invert(remove_hard)

    #         if not np.any(keep_hard):
    #             rm_dict.setdefault(q_cam_id, {})[q_track_id] = True
    #             continue

    #         sel_g_dis = dismat[index][order][keep]
    #         sel_g_track_ids = g_track_ids[order][keep]
    #         sel_g_cam_ids = g_cam_ids[order][keep]

    #         if sel_g_dis.size > 0:
    #             # pick the single best gallery candidate for this query
    #             best_idx       = int(np.argmin(sel_g_dis))
    #             best_g_track   = int(sel_g_track_ids[best_idx])
    #             best_g_cam     = sel_g_cam_ids[best_idx]
    #             best_dis       = float(sel_g_dis[best_idx])

    #             # record the proposal
    #             matches.append((int(q_cam_id), int(q_track_id), int(best_g_cam), best_g_track, best_dis))

    #     best_for_gallery = {}  # key: (g_cam, g_id) -> (dis, tuple_match)
    #     for (q_cam, q_id, g_cam, g_id, dis) in matches:
    #         key = (g_cam, g_id)
    #         if key not in best_for_gallery or dis < best_for_gallery[key][0]:
    #             best_for_gallery[key] = (dis, (q_cam, q_id, g_cam, g_id, dis))

    #     # winners = set of match tuples we keep
    #     winners = { tup for _, tup in best_for_gallery.values() }

    #     # filter matches down to winners only
    #     matches = [m for m in matches if m in winners]

    #     return matches, rm_dict


    def calc(self, dismat, q_track_ids, q_cam_ids, g_track_ids, g_cam_ids,
            q_times, g_times, q_entry_zn, q_exit_zn, g_entry_zn, g_exit_zn):
        """
        Returns:
            matches: List[(cam_id, track_id, matched_cam_id, matched_track_id, distance)]
            rm_dict: Dict[int, Dict[int, bool]]
            dismat:  np.ndarray (re-calculated distance matrix after KDE adjustments)
        """
        import numpy as np

        rm_dict = {}
        matches = []

        Q, G = dismat.shape
        indices = np.argsort(dismat, axis=1)              # (Q, G) per-row column orderings
        valid_mask = np.zeros_like(dismat, dtype=bool)    # True where pair survives filters

        warned_clm = False

        # ---------- 1) Per-row filtering to fill valid_mask (and update dismat via KDE) ----------
        for row, q_track_id in enumerate(q_track_ids):
            q_cam_id = q_cam_ids[row]
            q_time   = q_times[row]
            order    = indices[row]

            if self.clm is None:
                if not warned_clm:
                    self.logger.warning("Camera link model not found; skipping KDE time prior.")
                    warned_clm = True
                # Filter: same-cam or > threshold -> remove
                remove = (g_cam_ids[order] == q_cam_id) | (dismat[row][order] > self.dis_thre)
            else:
                # Apply KDE prior (this mutates dismat) and zone constraints
                dismat, directions = self._kde_filter(dismat, row, order, q_time, g_times, q_cam_id, g_cam_ids)
                zone_remove = self._zone_remove(
                    q_entry_zn[row], q_exit_zn[row],
                    g_entry_zn, g_exit_zn,
                    q_cam_id, g_cam_ids, order, directions
                )
                remove = (g_cam_ids[order] == q_cam_id) | (dismat[row][order] > self.dis_thre) | zone_remove

            keep = ~remove
            if np.any(keep):
                valid_mask[row, order[keep]] = True
            else:
                # No acceptable candidate for this query -> mark now
                rm_dict.setdefault(int(q_cam_id), {})[int(q_track_id)] = True

        # ---------- 2) Build cost matrix for assignment ----------
        LARGE = 1e6
        cost = np.where(valid_mask, dismat, LARGE)

        # ---------- 3) Hungarian assignment (globally optimal one-to-one) ----------
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception as e:
            # Fallback greedy if SciPy is unavailable
            self.logger.warning(f"Hungarian unavailable ({e}); falling back to greedy.")
            taken_r, taken_c = set(), set()
            pairs = []
            # sort valid pairs by ascending cost
            flat = [(r, c, cost[r, c]) for r in range(Q) for c in range(G) if valid_mask[r, c]]
            flat.sort(key=lambda x: x[2])
            for r, c, v in flat:
                if r not in taken_r and c not in taken_c:
                    taken_r.add(r); taken_c.add(c)
                    pairs.append((r, c))
            if pairs:
                row_ind = np.array([r for r, _ in pairs], dtype=int)
                col_ind = np.array([c for _, c in pairs], dtype=int)
            else:
                row_ind = np.array([], dtype=int)
                col_ind = np.array([], dtype=int)

        # ---------- 4) Build matches from assignment (respecting threshold & mask) ----------
        for r, c in zip(row_ind, col_ind):
            d = float(cost[r, c])
            if d >= LARGE:
                continue  # invalid pair assigned; ignore
            if d > float(self.dis_thre):
                continue  # defensively enforce threshold
            q_cam = int(q_cam_ids[r]); q_id = int(q_track_ids[r])
            g_cam = int(g_cam_ids[c]); g_id = int(g_track_ids[c])
            matches.append((q_cam, q_id, g_cam, g_id, d))
            # clear earlier "unmatched" mark if present
            if q_cam in rm_dict and q_id in rm_dict[q_cam]:
                del rm_dict[q_cam][q_id]
                if not rm_dict[q_cam]:
                    del rm_dict[q_cam]

        # ---------- 5) (Optional) mark queries that had valid pairs but weren't assigned ----------
        # If you want, uncomment to mark 'losers' as unmatched:
        # assigned_rows = set(row_ind.tolist())
        # for r in range(Q):
        #     if valid_mask[r].any() and r not in assigned_rows:
        #         q_cam = int(q_cam_ids[r]); q_id = int(q_track_ids[r])
        #         rm_dict.setdefault(q_cam, {})[q_id] = True

        # Return matches, rm_dict, and the updated dismat (with KDE adjustments)
        return matches, rm_dict
