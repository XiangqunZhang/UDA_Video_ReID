
import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.num_examples = 0
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        self.num_examples = len(self.list_of_examples)
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        # print(vid_list_file,"dsadad",self.list_of_examples)

    def next_batch(self, batch_size, flag):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        # for re-loading target data
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # dim: 2048 x frame#
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]  # ground truth (in words)
            classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))  # frame#
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # if different length, pad w/ zeros
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)  # zero-padding for shorter videos
        # mask = torch.zeros(4, 89, 8, dtype=torch.float)  # zero-padding for shorter videos

        # print(flag) #一s 一t
        # print(batch_input_tensor.size()) #torch.Size([1, 2048, 718])
        # print(batch_target_tensor.size()) #torch.Size([1, 2048, 718])
        # print(mask.size()) #torch.Size([1, 11, 718])
        # print(mask.sum())  #全0矩阵


        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        # print(mask.size())  # torch.Size([1, 11, 718])
        # print(mask) #每行都是到类别处 赋值为1

        return batch_input_tensor, batch_target_tensor, mask

class BatchGenerator_st(object):
    def __init__(self, num_classes,dataloder):
        self.num_examples = 0
        self.index = 0
        self.num_classes = num_classes
        self.dataloader1 = dataloder
        self.dataloader1_iter = iter(self.dataloader1)
        # self.cur = None



    def reset(self):
        self.index = 0
        self.dataloader1_iter = iter(self.dataloader1)

    def has_next(self):
        if self.index < len(self.dataloader1):
            return True
        return False

    # def has_next(self):
    #
    #     return False

    # def read_data(self, vid_list_file):
    #     file_ptr = open(vid_list_file, 'r')
    #     self.list_of_examples = file_ptr.read().split('\n')[:-1]
    #     self.num_examples = len(self.list_of_examples)
    #     file_ptr.close()
    #     random.shuffle(self.list_of_examples)
        # print(vid_list_file,"dsadad",self.list_of_examples)

    def next_batch1(self,flag):
        batch = next(self.dataloader1_iter)
        self.index += 1

        # for re-loading target data
        if flag == 'target' and self.index == len(self.dataloader1):
            self.reset()

        # batch[0] 1  2
        # self.cur = batch
        # if (self.index+1) % 30 == 0:
        #     print(self.index)
        # print(batch[1])

        return batch



    def next_batch2(self, batch_size,frames,fea,pids):

        batch_input = []
        batch_target = []
        # print(fea.size()[0])
        fea = fea.detach().cpu().numpy()
        pids = pids.detach().cpu().numpy()
        # print(fea)
        # print(pids)

        for i in range(fea.shape[0]):
            features = fea[i]  # dim: 2048 x frame#
            classes = np.full(frames,pids[i])  # ground truth (in indices)
            # classes = np.zeros(frames)  # ground truth (in indices)
            # for i in range(len(classes)):
            #     classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, ::1])
            batch_target.append(classes[::1])  #按1下采样

        length_of_sequences = list(map(len, batch_target))  # frame#
        batch_input_tensor = torch.zeros(batch_size, np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # if different length, pad w/ zeros
        batch_target_tensor = torch.ones(batch_size, max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(batch_size, self.num_classes, max(length_of_sequences), dtype=torch.float)  # zero-padding for shorter videos

        # print(flag) #一s 一t
        # print(batch_input_tensor.size()) #torch.Size([1, 2048, 718])
        # print(batch_target_tensor.size()) #torch.Size([1, 718])
        # print(mask.size()) #torch.Size([1, 11, 718])
        # print(mask) #torch.Size([1, 11, 718])


        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
    # def next_batch(self, batch_size, flag):
    #     batch = self.list_of_examples[self.index:self.index + batch_size]
    #     self.index += batch_size
    #
    #     # for re-loading target data
    #     if flag == 'target' and self.index == len(self.list_of_examples):
    #         self.reset()
    #
    #     batch_input = []
    #     batch_target = []
    #     for vid in batch:
    #         features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # dim: 2048 x frame#
    #         file_ptr = open(self.gt_path + vid, 'r')
    #         content = file_ptr.read().split('\n')[:-1]  # ground truth (in words)
    #         classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
    #         for i in range(len(classes)):
    #             classes[i] = self.actions_dict[content[i]]
    #         batch_input.append(features[:, ::self.sample_rate])
    #         batch_target.append(classes[::self.sample_rate])
    #
    #     length_of_sequences = list(map(len, batch_target))  # frame#
    #     batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # if different length, pad w/ zeros
    #     batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
    #     mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)  # zero-padding for shorter videos
    #
    #     # print(flag) #一s 一t
    #     # # print(batch_input_tensor.size()) #torch.Size([1, 2048, 718])
    #     # # print(batch_target_tensor.size()) #torch.Size([1, 2048, 718])
    #     # print(mask.size()) #torch.Size([1, 11, 718])
    #     # print(mask) #torch.Size([1, 11, 718])
    #
    #
    #     for i in range(len(batch_input)):
    #         batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
    #         batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
    #         mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
    #
    #     return batch_input_tensor, batch_target_tensor, mask

