# -*- coding: utf-8 -*-
# @File    : Generator.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/10/11

import os, random
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image, ImageDraw


def LoadImg(img_path):
    path = os.listdir(img_path)
    path.sort()
    imglist = []

    for i in range(len(path)):
        img = Image.open(osp.join(img_path, path[i]))
        imglist.append(img.copy())
        img.close()

    return imglist


def FindMatch(list_id, list1, list2):
    """

    :param list_id:
    :param list1:
    :param list2:
    :return:
    """
    index_pair = []
    for index, id in enumerate(list_id):
        index1 = list1.index(id)
        index2 = list2.index(id)
        index_pair.append(index1)
        index_pair.append(index2)

    return index_pair


class VideoData(object):

    def __init__(self, seq_id):
        self.img = LoadImg("data/MOT17/train/MOT17-{}-SDP/img1".format(seq_id))
        self.gt = np.loadtxt("data/MOT17/train/MOT17-{}-SDP/gt/gt.txt".format(seq_id), delimiter=',')

        self.ImageWidth = self.img[0].size[0]
        self.ImageHeight = self.img[0].size[1]

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def CurData(self, frame):
        data = self.gt[self.gt[:, 0] == (frame + 1)]    # 第一列是frame id，从0开始的

        return data

    def PreData(self, frame):
        DataList = []
        for i in range(5):  # fixme: hard-coded
            data = self.gt[self.gt[:, 0] == (frame + 1 - i)]
            DataList.append(data)

        return DataList

    def TotalFrame(self):

        return len(self.img)

    def CenterCoordinate(self, SingleLineData):
        x = (SingleLineData[2] + (SingleLineData[4] / 2)) / float(self.ImageWidth)
        y = (SingleLineData[3] + (SingleLineData[5] / 2)) / float(self.ImageHeight)

        return x, y

    def Appearance(self, data):
        """
        处理每一帧的图片，根据路人的bounding box裁剪对应的区域
        :param data:
        :return: appearance, list
        """
        appearance = []
        img = self.img[int(data[0, 0]) - 1]
        for i in range(data.shape[0]):
            crop = img.crop((int(data[i, 2]), int(data[i, 3]), int(data[i, 2]) + int(data[i, 4]),
                             int(data[i, 3]) + int(data[i, 5])))
            crop = self.transforms(crop)
            appearance.append(crop)

        return appearance

    def CurMotion(self, data):
        """

        :param data:
        :return: motion, list of 2-element tensors
        """
        motion = []
        for i in range(data.shape[0]):
            coordinate = torch.zeros([2])
            coordinate[0], coordinate[1] = self.CenterCoordinate(data[i])
            motion.append(coordinate)

        return motion

    def PreMotion(self, DataTuple):
        """

        :param DataTuple: list of motion information for each video sequence
        :return:
        """
        motion = []
        nameless = DataTuple[0]
        for i in range(nameless.shape[0]):
            coordinate = torch.zeros([5, 2])
            identity = nameless[i, 1]
            coordinate[4, 0], coordinate[4, 1] = self.CenterCoordinate(nameless[i])
            for j in range(1, 5):   # todo: 5 videos?
                unknown = DataTuple[j]
                if identity in unknown[:, 1]:  # if subject at frame t is also in the frame t-j
                    subject_prev_info = unknown[unknown[:, 1] == identity].squeeze()
                    coordinate[4 - j, 0], coordinate[4 - j, 1] = self.CenterCoordinate(
                            subject_prev_info)
                else:
                    coordinate[4 - j, :] = coordinate[5 - j, :]  # pad if the subject motion history < 5 (it does not happen)
            motion.append(coordinate)

        return motion

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1])

        return id

    def __call__(self, frame):
        """

        :param frame:
        :return:
        """
        assert frame >= 5 and frame < self.TotalFrame()
        cur = self.CurData(frame)
        pre = self.PreData(frame - 1)

        cur_crop = self.Appearance(cur)
        pre_crop = self.Appearance(pre[0])  # 只比较t和t-1状态的图片相似性

        cur_motion = self.CurMotion(cur)
        pre_motion = self.PreMotion(pre)    # list of (TIME_STEP, 2) tensors

        cur_id = self.GetID(cur)
        pre_id = self.GetID(pre[0])

        list_id = [x for x in pre_id if x in cur_id]  # list_id: 在前后两帧都出现的id
        index_pair = FindMatch(list_id, pre_id, cur_id)
        gt_matrix = np.zeros([len(pre_id), len(cur_id)])
#         修改
        for i in range(len(index_pair) // 2):
            gt_matrix[index_pair[2 * i], index_pair[2 * i + 1]] = 1

        return cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix


class ShipSequenceData(object):
    def __init__(self, seq_id: str):
        """

        :param seq_id: str, e.g., 55new.csv
        """
        csv_path = 'data/{}'.format(seq_id)
        df = pd.read_csv(csv_path)
        self.gt = df

        self.frame_len = len(df['Sample_Time'].unique())
        print(f'{self.frame_len} frames found in {csv_path}')

    def CurData(self, frame: int):
        data = self.gt[self.gt.iloc[:, 1] == frame]   # frame in the dataset starts from 0

        return data

    def PreData(self, frame: int):
        DataList = []
        for i in range(5):  # fixme: 5 is hard-coded
            data = self.gt[self.gt.iloc[:, 1] == (frame - i)]
            DataList.append(data)

        return DataList

    def TotalFrame(self):
        return self.frame_len

    def CurMotion(self, data):
        """

        :param data: pd.Series
        :return:
        """
        motion = []
        for i in range(data.shape[0]):
            coordinate = torch.zeros([2])
            coordinate[0], coordinate[1] = data.iloc[i, 2], data.iloc[i, 3]
            motion.append(coordinate)

        return motion

    def PreMotion(self, DataTuple):
        """

        :param DataTuple:
        :return:
        """
        motion = []
        t_minus_1 = DataTuple[0]
        for i in range(t_minus_1.shape[0]):
            coordinate = torch.zeros([5, 2])
            identity = t_minus_1.iloc[i, 0] # identity = ship id
            coordinate[4, 0], coordinate[4, 1] = t_minus_1.iloc[i, 2], t_minus_1.iloc[i, 3]
            for j in range(1, 5):
                unknown = DataTuple[j]
                identity_select_mask = unknown.iloc[:, 0] == identity
                if identity_select_mask.any():
                    subject_prev_info = unknown[identity_select_mask]
                    coordinate[4 - j, 0], coordinate[4 - j, 1] = (subject_prev_info.iloc[0, 2],
                                                                  subject_prev_info.iloc[0, 3])
                else:
                    coordinate[4 - j, :] = coordinate[5 - j, :]
            motion.append(coordinate)

        return motion

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data.iloc[i, 0])

        return id

    def __call__(self, frame: int):
        """
        :param frame: int, the id of frame
        :return: features
        """
        assert 5 <= frame < self.TotalFrame()

        cur = self.CurData(frame)  # 得到当前帧的数据
        pre = self.PreData(frame - 1)   # 得到之前的数据

        cur_crop = cur.iloc[:, 4:]  # 提取船舰本身的特征
        pre_crop = pre[0].iloc[:, 4:]

        cur_motion = self.CurMotion(cur)
        pre_motion = self.PreMotion(pre)

        cur_id = self.GetID(cur)
        pre_id = self.GetID(pre[0])

        list_id = [x for x in pre_id if x in cur_id]
        index_pair = FindMatch(list_id, pre_id, cur_id)
        gt_matrix = np.zeros([len(pre_id), len(cur_id)])
#         修改
        for i in range(len(index_pair) // 2):
            gt_matrix[index_pair[2 * i], index_pair[2 * i + 1]] = 1

        return cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix


class Generator(object):
    """
    For data loading.
    """
    def __init__(self, entirety=True, is_ship: bool = False):
        """

        :param entirety: bool, if to use all the videos in MOT17
        :param is_ship: bool, if to load ship dataset instead of videos
        """
        self.sequence = []

        if is_ship:
            self.SequenceID = ['55new_processed.csv', '2020new_processed.csv']

            self.vis_save_path = "ship_dataset/visualize"

            print("\n-------------------------- initialization --------------------------")
            for id in self.SequenceID:
                print("initializing sequence {} ...".format(id))
                self.sequence.append(ShipSequenceData(id))
                print("initialize {} done".format(id))
            print("------------------------------ done --------------------------------\n")
        else:
            if entirety:
                self.SequenceID = ["02", "04", "05", "09", "10", "11", "13"]
                # self.SequenceID = [ "05", "09"]
            else:
                self.SequenceID = ["09"]

            self.vis_save_path = "MOT17/visualize"

            print("\n-------------------------- initialization --------------------------")
            for id in self.SequenceID:
                print("initializing sequence {} ...".format(id))
                self.sequence.append(VideoData(id))
                print("initialize {} done".format(id))
            print("------------------------------ done --------------------------------\n")

    def visualize(self, seq_ID, frame, save_path=None):
        """

        :param seq_ID:
        :param frame:
        :param save_path:
        """
        if save_path is None:
            save_path = self.vis_save_path

        print("visualize sequence {}: frame {}".format(self.SequenceID[seq_ID], frame + 1))
        print("video solution: {} {}".format(self.sequence[seq_ID].ImageWidth, self.sequence[seq_ID].ImageHeight))
        cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix = self.sequence[seq_ID](frame)

        for i in range(len(cur_crop)):
            img = cur_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            # draw.text((0, 0), "id: {}\ncoord: {:3.2f}, {:3.2f}".format(int(cur_id[i]), cur_motion[i][0].item(),
            #                                                            cur_motion[i][1].item()), fill=(255, 0, 0))
            img.save(osp.join(save_path, "cur_crop_{}.png".format(str(i).zfill(2))))

        for i in range(len(pre_crop)):
            img = pre_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            # draw.text((0, 0), "id: {}\ncoord: {:3.2f}, {:3.2f}".format(int(pre_id[i]), pre_motion[i][4, 0].item(),
            #                                                            pre_motion[i][4, 1].item()), fill=(255, 0, 0))
            img.save(osp.join(save_path, "pre_crop_{}.png".format(str(i).zfill(2))))

        np.savetxt(osp.join(save_path, "gt_matrix.txt"), gt_matrix, fmt="%d")
        np.savetxt(osp.join(save_path, "pre_id.txt"), np.array(pre_id).transpose(), fmt="%d")
        np.savetxt(osp.join(save_path, "cur_id.txt"), np.array(cur_id).transpose(), fmt="%d")

    def __call__(self):
        """

        :return:
        """
        seq = random.choice(self.sequence)
        frame = random.randint(5, seq.TotalFrame() - 1) # For the LSTM module, we set the length of tracklet $L = 5$.
        cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix = seq(frame)

        return cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix
