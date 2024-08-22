import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import copy
import math
from torch.nn import init
import numpy as np


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #print("features:", features.size())

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            #print("labels", labels)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #print("mask", mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class attn_para(nn.Module):
    def __init__(self):
        super(attn_para, self).__init__()
        self.attn_para1 =  nn.Parameter(torch.FloatTensor([0,-1.55,-0.45]))

    def forward(self,x,y,z):
        loss = x * torch.exp(self.attn_para1[0]) + y * torch.exp(self.attn_para1[1]) + z * torch.exp(self.attn_para1[2])

        return loss

# definition of Gradient Reversal Layer
class GradRevLayer(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


# definition of Adversarial Domain Classifier (base part)
class AdvDomainClsBase(nn.Module):
    def __init__(self, in_feat, hidden_size, type_adv, args):
        super(AdvDomainClsBase, self).__init__()
        # ====== collect arguments ====== #
        self.num_f_maps = args.num_f_maps
        self.DA_adv_video = args.DA_adv_video
        self.pair_ssl = args.pair_ssl
        self.type_adv = type_adv

        # ====== main architecture ====== #
        self.fc1 = nn.Linear(in_feat, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, input_data, beta):
        feat = GradRevLayer.apply(input_data, beta)

        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)  

        return feat

# definition of Adversarial Domain Classifier (base part)
class AdvDomainClsBase_easy(nn.Module):
    def __init__(self, in_feat, hidden_size):
        super(AdvDomainClsBase_easy, self).__init__()

        # ====== main architecture ====== #
        self.fc1 = nn.Linear(in_feat, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, input_data, beta):
        feat = GradRevLayer.apply(input_data, beta)

        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)

        return feat


class MultiStageModel(nn.Module):
    def __init__(self, args, num_classes):
        super(MultiStageModel, self).__init__()
        # ====== collect arguments ====== #
        # this function only
        num_stages = args.num_stages
        num_layers = args.num_layers #10
        num_f_maps = args.num_f_maps #64
        dim_in = args.features_dim #2048
        method_centroid = args.method_centroid

        # cross-function
        self.use_target = args.use_target # usv
        self.multi_adv = args.multi_adv #【n,n】
        self.DA_adv_video = args.DA_adv_video #rev_grad
        self.ps_lb = args.ps_lb #soft
        self.use_attn = args.use_attn # domain_attn
        self.num_seg = args.num_seg #2
        self.pair_ssl = args.pair_ssl # all
        self.DA_ens = args.DA_ens #none
        self.SS_video = args.SS_video # none

        # ====== main architecture ====== #
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim_in, num_classes, self.DA_ens)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, self.DA_ens)) for s in range(num_stages-1)])




        # domain discriminators
        self.ad_net_base = nn.ModuleList()
        self.ad_net_base += [AdvDomainClsBase(num_f_maps, num_f_maps, 'frame', args)]
        self.ad_net_cls = nn.ModuleList()
        self.ad_net_cls += [nn.Linear(num_f_maps, 2)]

        # seq domain discriminators
        self.ad_net_seq_base = nn.ModuleList()
        self.ad_net_seq_base += [AdvDomainClsBase_easy(64, 64)]
        self.ad_net_seq_cls = nn.ModuleList()
        self.ad_net_seq_cls += [nn.Linear(64, 2)]

        # self.ad_net_seq_base = self.ad_net_seq_base.cuda()
        # self.ad_net_seq_cls = self.ad_net_seq_cls.cuda()

        self.bn = nn.BatchNorm1d(64)
        self.bn.apply(weights_init_kaiming)



    def predict_domain_seq(self, feat, beta_value):
        dim_feat = feat.size(1)
        feat = feat.reshape(-1, dim_feat)  # reshape to (batch x dim)
        out = self.ad_net_seq_cls[0](self.ad_net_seq_base[0](feat, beta_value))  # (batch, 2)
        # out = out.unsqueeze(1)  # (batch, 1, 2)

        return out


    def forward(self, x_s, x_t, mask_s, mask_t, beta, reverse):
        # forward source & target data at the same time
        # print("x_s",x_s.size())
        # print("x_t",x_t.size())
        # print("mask_s",mask_s.size())

        pred_source, prob_source, feat_source, pred_d_source, label_d_source, pred_source_2, prob_source_2,out_seq_d_fin_source,lb_out_seq_d_fin_source \
            = self.forward_domain(x_s, mask_s, 0, beta, reverse)
        pred_target, prob_target, feat_target, pred_d_target, label_d_target, pred_target_2, prob_target_2,out_seq_d_fin_target,lb_out_seq_d_fin_target  \
            = self.forward_domain(x_t, mask_t, 1, beta, reverse)

        # concatenate domain predictions & labels (frame-level)
        pred_d = torch.cat((pred_d_source, pred_d_target), 0)
        label_d = torch.cat((label_d_source, label_d_target), 0).long()

        pred_seq_d = torch.cat((out_seq_d_fin_source, out_seq_d_fin_target), 0)
        label_seq_d = torch.cat((lb_out_seq_d_fin_source, lb_out_seq_d_fin_target), 0).long()


        # self-supervised learning for videos

        return pred_source, prob_source, feat_source, pred_target, prob_target, feat_target, \
               pred_d, label_d, \
               pred_source_2, prob_source_2, pred_target_2, prob_target_2,pred_seq_d,label_seq_d

    def forward_domain(self, x, mask, domain_GT, beta, reverse):
        out_feat = self.stage1(x)  # out_feat: (batch, dim, frame#)
        out_feat_seq_raw = out_feat.mean(-1)
        out_feat_seq = self.bn(out_feat_seq_raw)

        # print('out_feat_seq',out_feat_seq.size())

        if reverse:  # reverse the gradient
            out_feat = GradRevLayer.apply(out_feat, beta[0])
            out_feat_seq = GradRevLayer.apply(out_feat_seq, beta[0])

        out = self.stage1.conv_out(out_feat)  # out: (batch, class#, frame#)
        out_2 = out.clone()

        prob = F.softmax(out, dim=1)  # prob: (batch, class#, frame#)
        prob_2 = F.softmax(out_2, dim=1)  # prob: (batch, class#, frame#)


        # compute domain predictions for single stage
        # out_d, lb_d = self.forward_stage(out_feat, beta, mask, domain_GT)
        out_d, lb_d,out_seq_d,lb_out_seq_d = self.forward_stage(out_feat,out_feat_seq, beta, mask, domain_GT)

        # store outputs
        outputs_feat = out_feat.unsqueeze(1)  # (batch, stage#, dim, frame#)

        outputs = out.unsqueeze(1)  # (batch, stage#, class#, frame#)
        probs = prob.unsqueeze(1)  # prob: (batch, stage#, class#, frame#)
        outputs_2 = out_2.unsqueeze(1)  # (batch, stage#, class#, frame#)
        probs_2 = prob_2.unsqueeze(1)  # prob: (batch, stage#, class#, frame#)

        outputs_d = out_d.unsqueeze(1)  # (batch x frame#, stage#, class#, 2)

        labels_d = lb_d.unsqueeze(1)  # (batch x frame#, stage#, class#)

        out_seq_d_fin = out_seq_d.unsqueeze(1)
        lb_out_seq_d_fin = lb_out_seq_d.unsqueeze(1)



        for s in self.stages:
            out_feat = s(prob)
            out_feat_seq_raw = out_feat.mean(-1)
            out_feat_seq = self.bn(out_feat_seq_raw)

            if reverse:  # reverse the gradient
                out_feat = GradRevLayer.apply(out_feat, beta[0])
                out_feat_seq = GradRevLayer.apply(out_feat_seq, beta[0])

            out = s.conv_out(out_feat)
            out_2 = out.clone()

            prob = F.softmax(out, dim=1)  # prob: (batch, class#, frame#)
            prob_2 = F.softmax(out_2, dim=1)  # prob: (batch, class#, frame#)

            # compute domain predictions for single stage
            out_d, lb_d,out_seq_d,lb_out_seq_d = self.forward_stage(out_feat,out_feat_seq, beta, mask, domain_GT)

            # store outputs
            outputs_feat = torch.cat((outputs_feat, out_feat.unsqueeze(1)), dim=1)

            outputs = torch.cat((outputs, out.unsqueeze(1)), dim=1)
            probs = torch.cat((probs, prob.unsqueeze(1)), dim=1)
            outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(1)), dim=1)
            probs_2 = torch.cat((probs_2, prob_2.unsqueeze(1)), dim=1)

            outputs_d = torch.cat((outputs_d, out_d.unsqueeze(1)), dim=1)

            labels_d = torch.cat((labels_d, lb_d.unsqueeze(1)), dim=1)

            out_seq_d_fin = torch.cat((out_seq_d_fin, out_seq_d.unsqueeze(1)), dim=1)
            lb_out_seq_d_fin = torch.cat((lb_out_seq_d_fin, lb_out_seq_d.unsqueeze(1)), dim=1)


        return outputs, probs, outputs_feat, outputs_d, labels_d, outputs_2, probs_2,out_seq_d_fin,lb_out_seq_d_fin

    def forward_stage(self, out_feat,out_feat_seq, beta, mask, domain_GT):
        # === Produce domain predictions === #
        # --- frame-level --- #
        # frame-wisely apply ad_net
        out_d = self.predict_domain_frame(out_feat, beta[0])  # out_d: (batch, class#, 2, frame#)

        out_seq_d = self.predict_domain_seq(out_feat_seq, beta[0])

        # # --- video-level --- #
        #pass

        # === Select valid frames + Generate domain labels === #
        # out_d, lb_d = self.select_masked(out_d, mask, domain_GT)
        out_d, lb_d,out_seq_d,lb_out_seq_d = self.select_masked(out_d,out_seq_d, mask, domain_GT)

        return out_d, lb_d,out_seq_d,lb_out_seq_d

    def predict_domain_frame(self, feat, beta_value):
        dim_feat = feat.size(1)
        num_frame = feat.size(2)
        # print('look domain frame pred',feat.size())
        feat = feat.transpose(1, 2).reshape(-1, dim_feat)    # reshape to (batch x frame#, dim)
        out = self.ad_net_cls[0](self.ad_net_base[0](feat, beta_value))  # (batch x frame#, 2)
        out = out.reshape(-1, num_frame, 2).transpose(1, 2)  # reshape back to (batch, 2, frame#)
        out = out.unsqueeze(1)  # (batch, 1, 2, frame#)

        return out


    # def select_masked(self, out_d, mask, out_d_video, domain_GT):
    def select_masked(self, out_d,out_seq_d, mask, domain_GT):
        # --- frame-level --- #
        # reshape --> (batch x frame#, ...)
        num_class_domain = out_d.size(1)
        out_d = out_d.transpose(2, 3).transpose(1, 2).reshape(-1, num_class_domain, 2)  # (batch x frame#, class#, 2)

        # select frames w/ mask + generate frame-level domain labels
        mask_frame = mask[:, 0, :].reshape(-1)  # (batch x frame#)
        mask_frame = mask_frame > 0
        out_d = out_d[mask_frame]  # (batch x valid_frame#, class#, 2)
        lb_d = torch.full_like(out_d[:, :, 0], domain_GT)  # lb_d: (batch x valid_frame#, class#)

        batch_size = out_seq_d.size(0)
        lb_out_seq_d = torch.full((batch_size, num_class_domain), domain_GT)
        # print(lb_out_seq_d)
        # print(lb_out_seq_d.size())

        return out_d, lb_d, out_seq_d, lb_out_seq_d



class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim_in, num_classes, DA_ens):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim_in, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x):
        out_feat = self.conv_1x1(x)
        for layer in self.layers:
            out_feat = layer(out_feat)
        return out_feat



class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)  
        return x + out

