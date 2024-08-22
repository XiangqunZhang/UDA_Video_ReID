import torch
import numpy as np
import random


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

    def next_batch1(self,flag):
        batch = next(self.dataloader1_iter)
        self.index += 1

        # for re-loading target data
        if flag == 'target' and self.index == len(self.dataloader1):
            self.reset()

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

        # print(flag)
        # print(batch_input_tensor.size()) #torch.Size([1, 2048, 718])
        # print(batch_target_tensor.size()) #torch.Size([1, 718])
        # print(mask.size()) #torch.Size([1, 11, 718])
        # print(mask) #torch.Size([1, 11, 718])

        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask

