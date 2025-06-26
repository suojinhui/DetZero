import copy
import random
from pathlib import Path

import numpy as np

from detzero_utils import common_utils

from detzero_refine.datasets.dataset import DatasetTemplate
from detzero_refine.utils.data_utils import sample_points, local_coords_transform
from detzero_refine.utils.geometry_augment import (augment_full_track,
                                                   augment_single_box,
                                                   test_time_augment)


class WaymoGeometryDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.query_num = self.dataset_cfg.get('QUERY_NUM', 3)
        self.query_pts_num = self.dataset_cfg.get('QUERY_POINTS_NUM', 256)
        self.memory_pts_num = self.dataset_cfg.get('MEMORY_POINTS_NUM', 4096)

        self.init_infos(self.mode)

    def extract_track_feature(self, data_info):
        traj_all = data_info['boxes_global'] # 整个轨迹上的box序列，这些边界框应该在其中一个框对应的坐标系下。
        score_all = data_info['score'] # 每个框的置信度
        frame_id_all = data_info['sample_idx'] # 帧 id
        pose_all = data_info['pose'] # 自车位姿
        pts_all = data_info['pts'] # 点云数据
        matched = data_info['matched'] # 是否匹配成功（这条轨迹中有的框无对应的GT与之匹配）
        mth_tk = data_info['matched_tracklet'] # 该轨迹是否与一条GT轨迹匹配上了
        state = data_info['state'] # 该目标的状态
        if data_info.get('gt_boxes_global', None) is not None:
            traj_gt_all = data_info['gt_boxes_global'] # gtbox

        # ---------------从整个序列中随机采样部分帧----------------

        # randomly sample the object track to increase data numbers 增加数据的多样性
        if self.training:
            traj_len = matched.sum()
            samples = random.sample(
                range(traj_len),
                random.randint(min(5, traj_len), traj_len)
            )

            # remove the data belonging to unmatched proposals
            #   and query the data based on sampled indices
            score = score_all[matched][samples]
            pose = pose_all[matched][samples]
            frm_id = frame_id_all[matched][samples]
            traj = traj_all[matched][samples]
            traj_gt = traj_gt_all[matched][samples]

            pts_mth = [pts_all[i] for i in range(len(traj_all)) if matched[i]]
            pts = [pts_mth[ind] for ind in samples]
        
        else:
            pts = pts_all
            traj = traj_all
            traj_gt = traj_gt_all
            pose = pose_all
            frm_id = frame_id_all
            score = score_all

        # --------------随机选取几帧作为查询------------------

        # randomly sample the query proposals
        if self.training:
            if self.query_num > len(traj):
                self.query_num = len(traj)
            query_idx = np.random.choice(len(traj), self.query_num, replace=False)
        else:
            # for inference, we select the query based on the confidence score
            query_idx = np.argsort(score)[::-1][:self.query_num]
            self.query_num = len(query_idx)
        
        # --------------将所有框的点云转化到当前同一的局部坐标系下--------

        # transform the object points to the corresponding box center frame
        pts = local_coords_transform(pts, traj)

        query_pts = [pts[ind].copy() for ind in query_idx] # 从这里看查询点云没有经过特征编码？
        query_box = np.array([traj[ind, :].copy() for ind in query_idx])
        gt_box = np.array([traj_gt[ind, :].copy() for ind in query_idx]) # 提取点云、预测box、gtbox

        # after all the transform, set box center and heading to 0
        query_box[:, [0, 1, 2, 6]] = 0
        gt_box[:, [0, 1, 2, 6]] = 0 # 对边界框做对齐，将xyz和朝向都设置为0


        # -------------对其中每个box的点云做随机的y方向翻转--------------
        # augment for each box of the object track
        if self.training and self.augment_single:
            pts = augment_single_box(pts)

        # -------------特征编码，将边界框的尺寸信息编码到点云中-----------

        # encode features for points of each proposal
        pts_new = []
        for idx, pts_per_box in enumerate(pts):
            pts_feat = []
            
            if 'placeholder' in self.encoding:
                pts_new = pts
                break
            
            if 'xyz' in self.encoding:
                pts_feat.append(pts_per_box[:, :3])

            if 'intensity' in self.encoding:
                pts_feat.append(pts_per_box[:, [3]])

            if 'p2s' in self.encoding:
                p2s_front = traj[idx][3:6]/2 - pts[idx][:, :3]
                p2s_back = traj[idx][3:6]/2 + pts[idx][:, :3]
                pts_feat.append(p2s_front)
                pts_feat.append(p2s_back)

            if 'score' in self.encoding: # 还有置信度信息
                pts_feat.append(np.array(score[idx]).repeat(pts_per_box.shape[0])[:, None])

            pts_new.append(np.concatenate(pts_feat, axis=1))

        pts = np.concatenate(pts_new, axis=0)

        # --------------对编码后点云做翻转、旋转、缩放-----------------

        # do augmentation for the whole object track
        if self.training and self.augment_full:
            pts, traj, query_pts, query_box, gt_box = augment_full_track( # 同时缩放了边界框的尺寸，缩放对整体时间序列的上的每个框是一致的
                pts, traj, query_pts, query_box, gt_box)

        # ---------------------执行点云随机采样------------------------
        # sample points for each proposal as the value feature
        pts = sample_points(pts, sample_num=self.memory_pts_num) # 采样4096个点 nxc
        # sample points for each query
        for ind in range(self.query_num):
            query_pts[ind] = sample_points(query_pts[ind], sample_num=self.query_pts_num) # 对每个查询选取256个点
        # query_points = np.array(query_pts)


        obj_info = {
            'sequence_name': data_info['sequence_name'],
            'frame': frm_id,
            'obj_id': data_info['obj_id'], # 目标id
            'obj_cls': self.class_map[data_info['name']], # 1/2/3 代表不同的类别
            'geo_query_num': self.query_num, # 查询数
            'geo_query_boxes': query_box, # 查询的box，已经统一同一个坐标系下了
            'geo_query_points': query_pts, # 查询Q的点云
            'geo_memory_points': pts, # K和V的点云特征
            'geo_trajectory': traj, # 时间序列上的所有框坐标尺寸
            'geo_score': score, # 置信度分数
            'gt_geo_query_boxes': gt_box, # 查询的gt框
            'gt_geo_trajectory': traj_gt, # 轨迹的gt框
            'pose': pose, # 自车位姿
            'state': state, # 状态
            'matched': matched, # 跟踪匹配结果
            'matched_tracklet': mth_tk # 该轨迹是否与一条GT轨迹匹配上了
        }

        return obj_info

    @staticmethod
    def tta_operator(data_dict):
        return test_time_augment(data_dict)

    @staticmethod
    def revert_to_each_frame(data_dict):
        res_list = []

        for i, pred_box in enumerate(data_dict['pred_boxes']):
            traj = data_dict['geo_trajectory'][i]
            box_preds_world = copy.deepcopy(traj)
            box_preds_world[:, 3:6] = pred_box[3:6][None, :].repeat(len(traj), axis=0)
            pose = data_dict['pose'][i]
            
            box_preds_lidar_per_frm = []
            for ind in range(len(pose)):
                r_t = np.linalg.inv(pose[ind])
                center = box_preds_world[[ind], :3]
                center = np.concatenate([center, np.ones((center.shape[0], 1))], axis=-1)
                center = center @ r_t.T
                heading = box_preds_world[[ind], [6]] + np.arctan2(r_t[1, 0], r_t[0, 0])
                
                box_lidar = np.concatenate([
                    center[:, :3],
                    box_preds_world[[ind], 3:6],
                    heading[None, :]], axis=-1)
                box_preds_lidar_per_frm.append(box_lidar)

            res_list.append(np.array(box_preds_lidar_per_frm))

        return res_list


    def generate_prediction_dicts(self, batch_dict, pred_dicts, single_pred_dict, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            single_pred_dict: dict, to save the result back

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_boxes = box_dict['pred_boxes']
            pred_dict = get_template_prediction(pred_boxes.shape[0])
            if pred_boxes.shape[0] == 0:
                print('-------------------ERROR--------------------')
                return pred_dict

            boxes_lidar = self.revert_to_each_frame(box_dict)

            pred_dict['boxes_lidar'] = boxes_lidar

            return boxes_lidar

        annos = []

        all_pred_res = generate_single_sample_dict(pred_dicts)
        for i in range(len(all_pred_res)):
            seq = batch_dict['sequence_name'][i]
            obj_id = batch_dict['obj_id'][i]
            
            # init the object result dict
            if seq not in single_pred_dict:
                single_pred_dict[seq] = {}
            
            single_pred_dict[seq][obj_id] = {
                'sequence_name': seq,
                'frame_id': [],
                'boxes_lidar': [],
                'score': [],
                'name': [],
                'pose': []
            }

            for idx, frm_id in enumerate(batch_dict['frame'][i]):
                single_pred_dict[seq][obj_id]['frame_id'].append(int(frm_id))
                single_pred_dict[seq][obj_id]['boxes_lidar'].append(all_pred_res[i][idx])
                single_pred_dict[seq][obj_id]['score'].append(batch_dict['geo_score'][i][idx])
                single_pred_dict[seq][obj_id]['name'].append(self.class_map[int(batch_dict['obj_cls'][i])])
                single_pred_dict[seq][obj_id]['pose'].append(pred_dicts['pose'][i][idx])

        return annos
