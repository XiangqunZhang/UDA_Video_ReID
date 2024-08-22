import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from sklearn.cluster import KMeans

from tqdm import tqdm

from models.vidDA_model import AdvDomainClsBase, AdvDomainClsBase_easy
from .losses import softmax_dice_loss, sigmoid_mse_loss, update_ema_variables
from .vidDA_batch_gen import BatchGenerator_st
from .vidDA_loss import *
from tensorboardX import SummaryWriter

from .utils import AverageMeter
from utils.losses import TripletLoss


class Trainer:
    def __init__(self, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_d = nn.CrossEntropyLoss(reduction='none')
        self.ce_dseq = nn.CrossEntropyLoss().cuda()

        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.temp_betain = None
        self.fea_consis_loss = torch.nn.MSELoss().cuda()
        # self.fea_consis_loss = TripletLoss(margin=0.3, distance='cosine')

        # seq domain discriminators
        self.ad_net_seq_base = nn.ModuleList()
        self.ad_net_seq_base += [AdvDomainClsBase_easy(2048, 2048)]
        self.ad_net_seq_cls = nn.ModuleList()
        self.ad_net_seq_cls += [nn.Linear(2048, 2)]

        self.ad_net_seq_base = self.ad_net_seq_base.cuda()
        self.ad_net_seq_cls = self.ad_net_seq_cls.cuda()

    def predict_domain_seq(self, feat, beta_value):
        dim_feat = feat.size(1)
        feat = feat.reshape(-1, dim_feat)  # reshape to (batch x dim)
        out = self.ad_net_seq_cls[0](self.ad_net_seq_base[0](feat, beta_value))  # (batch, 2)
        # out = out.unsqueeze(1)  # (batch, 1, 2)

        return out

    def adapt_weight(self, iter_now, iter_max_default, iter_max_input, weight_loss, weight_value=10.0, high_value=1.0, low_value=0.0):
        # affect adaptive weight value
        iter_max = iter_max_default
        if weight_loss < -1:
            iter_max = iter_max_input

        high = high_value
        low = low_value
        weight = weight_value
        p = float(iter_now) / iter_max
        adaptive_weight = (2. / (1. + np.exp(-weight * p)) - 1) * (high-low) + low
        return adaptive_weight

    def compute_group_labels(self, features, features_ema, num_clusters):
        # 将特征转换为NumPy数组并连接起来
        features_combined = np.concatenate([features.detach().cpu().numpy(), features_ema.detach().cpu().numpy()])

        # 使用K-means算法进行聚类
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features_combined)

        # 获取每个特征所属的簇索引
        labels = kmeans.labels_

        # 将标签转换为PyTorch张量并放在CUDA上
        group_labels = torch.from_numpy(labels).cuda()

        # 计算每个特征样本与其所在类簇中心点特征的距离
        cluster_centers = kmeans.cluster_centers_
        distances = np.linalg.norm(features_combined - cluster_centers[labels], axis=1)
        avg_distance = np.mean(distances)

        return group_labels[:len(features)], group_labels[len(features):], avg_distance

    def zero_one_loss(self, group_label1, group_label2):
        """
        计算0-1损失函数，用于比较两组标签张量。

        参数:
        - group_label1: 第一组标签张量，类型为torch.Tensor，要求在GPU上。
        - group_label2: 第二组标签张量，类型为torch.Tensor，要求在GPU上。

        返回:
        - zero_one_loss: 0-1损失值，类型为浮点数。
        """
        assert group_label1.shape == group_label2.shape, "标签张量的形状不一致"

        num_samples = group_label1.size(0)
        misclassified = torch.sum(group_label1 != group_label2).item()

        zero_one_loss = misclassified / num_samples

        return zero_one_loss

    def train(self,ema_model,attn_model,attn_optimizer, model, model_dir, results_dir, device, args,epoch, ori_model,ori_optimizer, criterions,trainloader,t_trainloader, source_num_classes, target_num_classes, use_gpu):

        model.train()
        ori_model.train()

        s_iter_batch = len(trainloader)
        # print("source_batch_count:", len(trainloader))
        # print("target_batch_count:", len(t_trainloader))
        # print("source_batch_count:", source_num_classes)
        # print("target_batch_count:", target_num_classes)

        batch_gen_source = BatchGenerator_st(source_num_classes, trainloader)
        batch_gen_target = BatchGenerator_st(target_num_classes, t_trainloader)

        # ====== collect arguments ====== #
        verbose = args.verbose
        num_epochs = args.num_epochs
        batch_size = args.bS
        num_f_maps = args.num_f_maps
        learning_rate = args.lr
        alpha = args.alpha
        tau = args.tau
        use_target = args.use_target
        ratio_source = args.ratio_source
        ratio_label_source = args.ratio_label_source
        resume_epoch = args.resume_epoch
        # tensorboard
        use_tensorboard = args.use_tensorboard
        epoch_embedding = args.epoch_embedding
        stage_embedding = args.stage_embedding
        num_frame_video_embedding = args.num_frame_video_embedding

        # adversarial loss
        DA_adv = args.DA_adv
        DA_adv_video = args.DA_adv_video
        iter_max_beta_user = args.iter_max_beta
        place_adv = args.place_adv
        beta = args.beta

        # multi-class adversarial loss
        multi_adv = args.multi_adv
        weighted_domain_loss = args.weighted_domain_loss
        ps_lb = args.ps_lb

        # semantic loss
        method_centroid = args.method_centroid
        DA_sem = args.DA_sem
        place_sem = args.place_sem
        ratio_ma = args.ratio_ma
        gamma = args.gamma
        iter_max_gamma_user = args.iter_max_gamma

        # entropy loss
        DA_ent = args. DA_ent
        place_ent = args.place_ent
        mu = args.mu

        # discrepancy loss
        DA_dis = args.DA_dis
        place_dis = args.place_dis
        nu = args.nu
        iter_max_nu_user = args.iter_max_nu

        # ensemble loss
        DA_ens = args.DA_ens
        place_ens = args.place_ens
        dim_proj = args.dim_proj

        # self-supervised learning for videos
        SS_video = args.SS_video
        place_ss = args.place_ss
        eta = args.eta

        # multi-GPU
        if args.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.train()
        ema_model.train()
        model.to(device)

        torch.save(model.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + ".model")

        if resume_epoch > 0:
            model.load_state_dict(torch.load(model_dir + "/epoch-" + str(resume_epoch) + ".model"))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # determine batch size
        num_iter_epoch = 1

        acc_best_source = 0.0  # store the best source acc
        acc_best_target = 0.0  # store the best target acc
        if use_tensorboard:
            pass

        epoch_loss = 0
        epoch_consis_loss = 0
        epoch_consis_loss_wo = 0
        epoch_reid_loss = 0
        correct_source = 0
        total_source = 0
        correct_target = 0
        total_target = 0
        iter_batch = 0  #iter_可以理解为


        start_iter = epoch * s_iter_batch
        iter_max_default = num_epochs * s_iter_batch  # affect adaptive weight value

        # initialize the embedding
        batch_xent_loss = AverageMeter()
        batch_htri_loss = AverageMeter()
        batch_info_loss = AverageMeter()
        batch_loss = AverageMeter()
        batch_corrects = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # start ori training!
        end = time.time()
        pd = tqdm(total=len(trainloader), ncols=120, leave=False)
        while(batch_gen_source.has_next()):
            pd.set_postfix({'Acc': '{:>7.2%}'.format(batch_corrects.avg), })
            pd.update(1)

            ori_model.train()
            s_batch = batch_gen_source.next_batch1("source")
            t_batch = batch_gen_target.next_batch1("target")

            vids_s = s_batch[0]
            pids_s = s_batch[1]
            vids_t = t_batch[0]
            pids_t = t_batch[1]

            if use_gpu:
                vids_s, pids_s, vids_t, pids_t = vids_s.cuda(), pids_s.cuda(), vids_t.cuda(), pids_t.cuda()

            # measure data loading time
            data_time.update(time.time() - end)

            # zero the parameter gradients
            ori_optimizer.zero_grad()

            # forward
            if 'infonce' in args.losses:
                y, f, x,s_f = ori_model(vids_s)
                xent_loss = criterions['xent'](y, pids_s)
                htri_loss = criterions['htri'](f, pids_s)
                info_loss = criterions['infonce'](x)
                reid_loss = xent_loss + htri_loss + 0.001 * info_loss
            else:
                y, f,s_f = ori_model(vids_s)
                # combine hard triplet loss with cross entropy loss
                xent_loss = criterions['xent'](y, pids_s)
                htri_loss = criterions['htri'](f, pids_s)
                reid_loss = xent_loss + htri_loss
                info_loss = htri_loss * 0


            t_y_pseudo,t_f_d1,_,t_f = ori_model(vids_t)

            # ema
            # noise = torch.clamp(torch.randn_like(vids_t) * 0.1, -0.2, 0.2)
            # ema_inputs = vids_t + noise

            ema_inputs = vids_t
            with torch.no_grad():
                t_y_pseudo_ema, t_f_ema_d1,_, mid_t_f_ema = ema_model(ema_inputs)

            # # 计算group标签和group_dis_loss
            group_labels_features, group_labels_features_ema, group_dis_loss = self.compute_group_labels(t_f_d1, t_f_ema_d1, 16)
            # # 计算0-1损失
            zero_one_loss = self.zero_one_loss(group_labels_features, group_labels_features_ema)

            # start training
            # while batch_gen_source.has_next():
            #
            # adaptive weight for adversarial loss
            iter_now = iter_batch + start_iter
            adaptive_beta_0 = self.adapt_weight(iter_now, iter_max_default, iter_max_beta_user[0], beta[0])
            adaptive_beta_1 = self.adapt_weight(iter_now, iter_max_default, iter_max_beta_user[1], beta[1]) / 10.0
            adaptive_gamma = self.adapt_weight(iter_now, iter_max_default, iter_max_gamma_user, gamma)
            adaptive_nu = self.adapt_weight(iter_now, iter_max_default, iter_max_nu_user, nu)
            beta_in_0 = adaptive_beta_0 if beta[0] < 0 else beta[0]
            beta_in_1 = adaptive_beta_1 if beta[1] < 0 else beta[1]
            beta_in = [beta_in_0, beta_in_1]
            self.temp_betain = beta_in
            gamma_in = adaptive_gamma if gamma < 0 else gamma
            nu_in = adaptive_nu if nu < 0 else nu

            # ====== Feed-forward data ====== #
            # prepare inputs
            input_source, label_source, mask_source = batch_gen_source.next_batch2(args.bS,args.seq_len,s_f.detach(),pids_s.detach())
            input_source, label_source, mask_source = input_source.to(device), label_source.to(device), mask_source.to(device)

            # drop some source frames (including labels) for semi-supervised learning
            input_source, label_source, mask_source = self.ctrl_video_length(input_source, label_source, mask_source, ratio_source)

            # drop source labels only
            label_source_new, mask_source_new = self.ctrl_video_label_length(label_source, mask_source, ratio_label_source)

            input_target, label_target, mask_target = batch_gen_target.next_batch2(args.bS,args.seq_len,t_f.detach(),pids_t.detach())
            input_target, label_target, mask_target = input_target.to(device), label_target.to(device), mask_target.to(device)

            # forward-pass data
            # label: (batch, frame#)
            # pred: (batch, stage#, class#, frame#)
            # feat: (batch, stage#, dim, frame#)
            # pred_d: (batch x frame#, stage#, class#, 2)
            # pred_d_video: (batch x seg#, stage#, 2)
            pred_source, prob_source, feat_source, pred_target, prob_target, feat_target, \
            pred_d, label_d, \
            pred_source_2, prob_source_2, pred_target_2, prob_target_2,pred_seq_d,label_seq_d \
                = model(input_source, input_target, mask_source, mask_target, beta_in, reverse=False)


            num_stages = pred_source.shape[1]

            # ------ Classification loss ------ #
            loss = 0
            for s in range(num_stages):
                p = pred_source[:, s, :, :]  # select one stage --> (batch, class#, frame#)
                loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), label_source_new.view(-1))
                loss += alpha * torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=tau ** 2) * mask_source_new[:, :, 1:])

            # ====== Domain Adaptation ====== #
            if use_target != 'none':
                # 有
                num_class_domain = pred_d.size(2)

                for s in range(num_stages):
                    # --- select data for the current stage --- #
                    # masking class prediction
                    pred_select_source, prob_select_source, prob_select_source_2, feat_select_source, label_select_source, classweight_stage_select_source \
                        = self.select_data_stage(s, pred_source, prob_source, prob_source_2, feat_source, label_source)

                    pred_select_target, prob_select_target, prob_select_target_2, feat_select_target, label_select_target, classweight_stage_select_target \
                        = self.select_data_stage(s, pred_target, prob_target, prob_target_2, feat_target, label_target)

                    # masking domain prediction
                    pred_d_stage, label_d_stage \
                        = self.select_data_domain_stage(s, pred_d, label_d)

                    pred_seq_d_stage, label_seq_d_stage \
                        = self.select_data_domain_stage2(s, pred_seq_d, label_seq_d)

                    # concatenate class probability masks
                    classweight_stage = torch.cat((classweight_stage_select_source, classweight_stage_select_target), 0)

                    # ------ Adversarial loss ------ #
                    if DA_adv == 'rev_grad':
                        if place_adv[s] == 'Y':
                            # calculate loss
                            loss_adv = 0
                            for c in range(num_class_domain):
                                pred_d_class = pred_d_stage[:, c, :]  # (batch x frame#, 2)
                                label_d_class = label_d_stage[:, c]  # (batch x frame#)
                                loss_adv_class = self.ce_d(pred_d_class, label_d_class)
                                loss_adv += loss_adv_class.mean()

                            loss += loss_adv

                            # 加上seq的advloss
                            label_seq_d_stage = torch.squeeze(label_seq_d_stage).cuda()
                            loss_adv_seq = self.ce_dseq(pred_seq_d_stage, label_seq_d_stage)
                            loss += loss_adv_seq

                    # ------ Entropy loss ------ #
                    if DA_ent == 'target':
                        pass
                    elif DA_ent == 'attn':
                        if place_ent[s] == 'Y':
                            # calculate loss
                            loss_ent = 0
                            for c in range(num_class_domain):
                                pred_d_class = pred_d_stage[:, c, :]  # (batch x frame#, 2)

                                loss_ent_class = attentive_entropy(torch.cat((pred_select_source, pred_select_target), 0), pred_d_class)
                                if weighted_domain_loss == 'Y' and multi_adv[1] == 'Y':  # weighted by class prediction
                                    if ps_lb == 'soft':  #this this!!!
                                        loss_ent_class *= classweight_stage[:, c].detach()

                                loss_ent += loss_ent_class.mean()
                            loss += mu * loss_ent


            # training
            optimizer.zero_grad()

            loss.backward(retain_graph=True)

            #
            consis_loss = zero_one_loss * 10 + group_dis_loss

            ori_loss = attn_model(reid_loss, loss, consis_loss)

            ori_loss.backward()
            optimizer.step()
            ori_optimizer.step()
            attn_optimizer.step()

            update_ema_variables(ori_model,ema_model,0.99,iter_batch)


            # statistics
            _, preds = torch.max(y.data, 1)
            batch_corrects.update(torch.sum(preds == pids_s.data).float() / pids_s.size(0), pids_s.size(0))
            batch_xent_loss.update(xent_loss.item(), pids_s.size(0))
            batch_htri_loss.update(htri_loss.item(), pids_s.size(0))
            batch_info_loss.update(info_loss.item(), pids_s.size(0))
            batch_loss.update(loss.item(), pids_s.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            epoch_loss += loss.item()  # record the epoch loss
            epoch_consis_loss += consis_loss.item()  # record the epoch_consis loss
            # epoch_consis_loss += consis_loss # record the epoch_consis loss
            # epoch_consis_loss_wo += consistency_loss.item()
            epoch_reid_loss += reid_loss.item()

            # prediction
            _, pred_id_source = torch.max(pred_source[:, -1, :, :].data, 1)  # predicted indices (from the last stage)
            correct_source += ((pred_id_source == label_source_new).float() * mask_source_new[:, 0, :].squeeze(1)).sum().item()
            total_source += torch.sum(mask_source_new[:, 0, :]).item()
            _, pred_id_target = torch.max(pred_target[:, -1, :, :].data, 1)  # predicted indices (from the last stage)
            correct_target += ((pred_id_target == label_target).float() * mask_target[:, 0, :].squeeze(1)).sum().item()
            total_target += torch.sum(mask_target[:, 0, :]).item()

            iter_batch += 1

        # print(attn_model.state_dict())
        #
        if((epoch+1)%10 == 0):
            torch.save(model.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + ".opt")

        acc_epoch_source = float(correct_source) / total_source
        acc_epoch_target = float(correct_target) / total_target
        # update the "best" model (best training acc)
        if acc_epoch_source > acc_best_source:
            acc_best_source = acc_epoch_source
            torch.save(model.state_dict(), model_dir + "/acc_best_source.model")
            torch.save(optimizer.state_dict(), model_dir + "/acc_best_source.opt")

        if acc_epoch_target > acc_best_target:
            acc_best_target = acc_epoch_target
            torch.save(model.state_dict(), model_dir + "/acc_best_target.model")
            torch.save(optimizer.state_dict(), model_dir + "/acc_best_target.opt")

        if verbose:
            print("[epoch %d]: epoch loss = %f,   acc (S) = %f,   acc (T) = %f,   beta = (%f, %f),   nu = %f, epoch_consis_loss = %f, epoch_reid_loss = %f"% (epoch + 1, epoch_loss / num_iter_epoch, acc_epoch_source, acc_epoch_target, beta_in[0], beta_in[1], nu_in, epoch_consis_loss, epoch_reid_loss))  # uncomment for debugging
            # print("epoch_consis_loss:",epoch_consis_loss)
            # print("epoch_reid_loss:",epoch_reid_loss)

        pd.close()


        print('Epoch{0} '
              'Time:{batch_time.sum:.1f}s '
              'Data:{data_time.sum:.1f}s '
              'Loss:{loss.avg:.4f} '
              'Xent:{xent.avg:.4f} '
              'Htri:{htri.avg:.4f} '
              'Acc:{acc.avg:.2%} '.format(
            epoch + 1, batch_time=batch_time,
            data_time=data_time, loss=batch_loss,
            xent=batch_xent_loss, htri=batch_htri_loss,
            acc=batch_corrects))


    def select_data_stage(self, s, pred, prob, prob_2, feat, label):
        dim_feat = feat.size(2)

        # features & prediction
        feat_stage = feat[:, s, :, :]  # select one stage --> (batch, dim, frame#)
        feat_frame = feat_stage.transpose(1, 2).reshape(-1, dim_feat)
        pred_stage = pred[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        pred_frame = pred_stage.transpose(1, 2).reshape(-1, self.num_classes)
        prob_stage = prob[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        prob_frame = prob_stage.transpose(1, 2).reshape(-1, self.num_classes)
        prob_2_stage = prob_2[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        prob_2_frame = prob_2_stage.transpose(1, 2).reshape(-1, self.num_classes)

        # select the masked frames & labels
        label_vector = label.reshape(-1).clone()
        feat_select = feat_frame[label_vector != -100]
        pred_select = pred_frame[label_vector != -100]
        label_select = label_vector[label_vector != -100]
        prob_select = prob_frame[label_vector != -100]
        prob_2_select = prob_2_frame[label_vector != -100]

        # class probability as class weights
        classweight_stage = prob[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        classweight_stage = classweight_stage.transpose(1, 2).reshape(-1, self.num_classes)  # (batch x frame#, class#)

        # mask frames
        classweight_stage_select = classweight_stage[label_vector != -100]

        return pred_select, prob_select, prob_2_select, feat_select, label_select, classweight_stage_select

    def select_data_domain_stage(self, s, pred_d, label_d):

        # domain predictions & labels (frame-level)
        pred_d_select = pred_d[:, s, :, :]  # select one stage --> (batch x frame#, class#, 2)
        label_d_select = label_d[:, s, :]  # select one stage --> (batch x frame#, class#)

        # # domain predictions & labels (video-level)
        # pred_d_select_seg = pred_d_video[:, s, :]  # select one stage --> (batch x seg#, 2)
        # label_d_select_video = label_d_video[:, s]  # select one stage --> (batch x seg#)

        # return pred_d_select, pred_d_select_seg, label_d_select, label_d_select_video
        return pred_d_select, label_d_select

    def select_data_domain_stage2(self, s, pred_d, label_d):

        # domain predictions & labels (frame-level)
        pred_d_select = pred_d[:, s, :]  # select one stage --> (batch x frame#, class#, 2)
        label_d_select = label_d[:, s, :]  # select one stage --> (batch x frame#, class#)

        # # domain predictions & labels (video-level)
        # pred_d_select_seg = pred_d_video[:, s, :]  # select one stage --> (batch x seg#, 2)
        # label_d_select_video = label_d_video[:, s]  # select one stage --> (batch x seg#)

        # return pred_d_select, pred_d_select_seg, label_d_select, label_d_select_video
        return pred_d_select, label_d_select

    def ctrl_video_length(self, input_data, label, mask, ratio_length):
        # shapes:
        # input_data: (batch, dim, frame#)
        # label: (batch, frame#)
        # mask: (batch, class#, frame#)

        # get the indices of the frames to keep
        num_frame = input_data.size(-1)  # length of video
        num_frame_drop = (1 - ratio_length) * num_frame
        id_drop = np.floor(np.linspace(0, num_frame-1, num_frame_drop)).tolist()
        id_keep = list(set(range(num_frame)) - set(id_drop))
        id_keep = torch.tensor(id_keep).long()
        if input_data.get_device() >= 0:
            id_keep = id_keep.to(input_data.get_device())

        # filter the inputs w/ the above indices
        input_data_filtered = input_data[:, :, id_keep]
        label_filtered = label[:, id_keep]
        mask_filtered = mask[:, :, id_keep]

        return input_data_filtered, label_filtered, mask_filtered

    def ctrl_video_label_length(self, label, mask, ratio_length):
        # shapes:
        # label: (batch, frame#)
        # mask: (batch, class#, frame#)
        mask_new = mask.clone()
        label_new = label.clone()

        # get the indices of the frames to keep
        num_frame = mask.size(-1)  # length of video
        num_frame_drop = (1 - ratio_length) * num_frame
        id_drop = np.floor(np.linspace(0, num_frame-1, num_frame_drop)).tolist()
        id_drop = torch.tensor(id_drop).long()
        if mask.get_device() >= 0:
            id_drop = id_drop.to(mask.get_device())

        # assign 0 to the above indices
        mask_new[:, :, id_drop] = 0
        label_new[:, id_drop] = -100  # class id -100 won't be calculated in cross-entropy

        return label_new, mask_new
