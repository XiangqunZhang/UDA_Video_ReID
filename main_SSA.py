from __future__ import print_function, absolute_import
import os
import sys
import time
import random
import datetime
import argparse

import faiss
import numpy as np
import os.path as osp

from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from models import init_model
from models.vidDA_model import attn_para, MultiStageModel, attn_para00
from utils.losses import TripletLoss, InfoNce
# from utils.sstda_train_newcluloss import Trainer
from utils.vidDA_train import Trainer
from utils.utils import AverageMeter, Logger, save_checkpoint, print_time
from utils.eval_metrics import evaluate,evaluate_with_clothes
from utils.samplers import RandomIdentitySampler,RandomIdentitySampler_vccvid
from utils import data_manager, ramps
from utils.video_loader import ImageDataset, VideoDataset, VideoDatasetInfer, VideoDataset2, VideoDataset1, \
    VideoDatasetInfer1, Preprocessor_videoDataset1

from utils.faiss_rerank import compute_jaccard_distance
import torch.nn.functional as F
from torch import nn, autograd

import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bbbbb_ssssize = 32
eeeepoch = 150

parser = argparse.ArgumentParser(description='Train video model')
# Datasets
parser.add_argument('--root', type=str, default='/HDD3/zxq')
# parser.add_argument('--td', type=str, default='ilidsvid')
parser.add_argument('--td', type=str, default='ccvid')
parser.add_argument('-d', '--dataset', type=str, default='v3dgait',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# Augment
parser.add_argument('--sample_stride', type=int, default=8, help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=150, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=bbbbb_ssssize, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=[40, 80, 120], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--distance', type=str, default='cosine', help="euclidean or consine")
parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
parser.add_argument('--losses', default=['xent', 'htri'], nargs='+', type=str, help="losses")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='sinet', help="c2resnet50, nonlocalresnet50")
parser.add_argument('--pretrain', action='store_true', help="load params form pretrain model on kinetics")
parser.add_argument('--pretrain_model_path', type=str, default='', metavar='PATH')
# Miscs
parser.add_argument('--seed', type=int, default=1538574472, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=5,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', '--sd', type=str, default='log_default')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='0,1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--all_frames', action='store_true', help="evaluate with all frames ?")
parser.add_argument('--seq_len', type=int, default=4,
                    help="number of images to sample in a tracklet")
parser.add_argument('--note', type=str, default='', help='additional description of this command')

#SSA args
# architecture
parser.add_argument('--num_stages', default=4, type=int, help='stage number')
parser.add_argument('--num_layers', default=10, type=int, help='layer number in each stage')
parser.add_argument('--num_f_maps', default=64, type=int, help='embedded feat. dim.')
parser.add_argument('--features_dim', default=2048, type=int, help='input feat. dim.')
parser.add_argument('--DA_adv', default='rev_grad', type=str, help='adversarial loss (none | rev_grad)')
parser.add_argument('--DA_adv_video', default='rev_grad_ssl', type=str, help='video-level adversarial loss (none | rev_grad | rev_grad_ssl | rev_grad_ssl_2)')
parser.add_argument('--pair_ssl', default='all', type=str, help='pair-feature methods for SSL-DA (all | adjacent)')
parser.add_argument('--num_seg', default=2, type=int, help='segment number for each video')
parser.add_argument('--place_adv', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_adv) == num_stages')
parser.add_argument('--multi_adv', default=['N', 'N'], type=str, nargs="+",
                    metavar='N', help='separate weights for domain discriminators')
parser.add_argument('--weighted_domain_loss', default='Y', type=str, help='weighted domain loss for class-wise domain discriminators')
parser.add_argument('--ps_lb', default='soft', type=str, help='pseudo-label type (soft | hard)')
parser.add_argument('--source_lb_weight', default='pseudo', type=str, help='label type for source data weighting (real | pseudo)')
parser.add_argument('--method_centroid', default='none', type=str, help='method to get centroids (none | prob_hard)')
parser.add_argument('--DA_sem', default='mse', type=str, help='metric for semantic loss (none | mse)')
parser.add_argument('--place_sem', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_sem) == num_stages')
parser.add_argument('--ratio_ma', default=0.7, type=float, help='ratio for moving average centroid method')
parser.add_argument('--DA_ent', default='attn', type=str, help='entropy-related loss (none | target | attn)')
parser.add_argument('--place_ent', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_ent) == num_stages')
parser.add_argument('--use_attn', type=str, default='domain_attn', choices=['none', 'domain_attn'], help='attention mechanism')
parser.add_argument('--DA_dis', type=str, default='none', choices=['none', 'JAN'], help='discrepancy method for DA')
parser.add_argument('--place_dis', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_dis) == num_stages')
parser.add_argument('--DA_ens', type=str, default='none', choices=['none', 'MCD', 'SWD'], help='ensemble method for DA')
parser.add_argument('--place_ens', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_ens) == num_stages')
parser.add_argument('--SS_video', type=str, default='none', choices=['none', 'VCOP'], help='video-based self-supervised learning method')
parser.add_argument('--place_ss', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                    metavar='N', help='len(place_ss) == num_stages')
# config & setting
parser.add_argument('--path_data', default='data/')
parser.add_argument('--path_model', default='models/')
parser.add_argument('--path_result', default='results/')
parser.add_argument('--action', default='train')
parser.add_argument('--use_target', default='uSv', choices=['none', 'uSv'])
parser.add_argument('--split_target', default='0', help='split for target data (0: no additional split for target)')
parser.add_argument('--ratio_source', default=1, type=float, help='percentage of total length to use for source data')
parser.add_argument('--ratio_label_source', default=1, type=float, help='percentage of labels to use for source data (after previous processing)')
parser.add_argument('--st_dataset', default="gtea")
parser.add_argument('--split', default='1')
# hyper-parameters
# parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--bS', default=bbbbb_ssssize, type=int, help='batch size')
parser.add_argument('--alpha', default=0.15, type=float, help='weighting for smoothing loss')
parser.add_argument('--tau', default=4, type=float, help='threshold to truncate smoothing loss')
parser.add_argument('--beta', default=[-2, -2], type=float, nargs="+", metavar='M', help='weighting for adversarial loss & ensemble loss ([frame-beta, video-beta])')
# parser.add_argument('--iter_max_beta', new batch，default=[2000, 1400], type=float, nargs="+", metavar='M', help='for adaptive beta ([frame-beta, video-beta])')
parser.add_argument('--iter_max_beta', default=[20000, 20000], type=float, nargs="+", metavar='M', help='for adaptive beta ([frame-beta, video-beta])')
parser.add_argument('--st_gamma', default=-2, type=float, help='weighting for semantic loss')
parser.add_argument('--iter_max_gamma', default=2000, type=float, help='for adaptive gamma')
parser.add_argument('--mu', default=0.01, type=float, help='weighting for entropy loss')
parser.add_argument('--nu', default=-2, type=float, help='weighting for the discrepancy loss')
parser.add_argument('--eta', default=1, type=float, help='weighting for the self-supervised loss')
parser.add_argument('--iter_max_nu', default=1000, type=float, metavar='M', help='for adaptive nu')
parser.add_argument('--dim_proj', default=128, type=int, help='projection dimension for SWD')
# runtime
parser.add_argument('--num_epochs', default=eeeepoch, type=int)
parser.add_argument('--verbose', default=True, action="store_true")
parser.add_argument('--use_best_model', type=str, default='none', choices=['none', 'source', 'target'], help='save best model')
parser.add_argument('--multi_gpu', default=False, action="store_true")
parser.add_argument('--resume_epoch', default=0, type=int)
# tensorboard
parser.add_argument('--use_tensorboard', default=False, action='store_true')
parser.add_argument('--epoch_embedding', default=50, type=int, help='select epoch # to save embedding (-1: all epochs)')
parser.add_argument('--stage_embedding', default=-1, type=int, help='select stage # to save embedding (-1: last stage)')
parser.add_argument('--num_frame_video_embedding', default=50, type=int, help='number of sample frames per video to store embedding')

args = parser.parse_args()

def specific_params(args):
    if args.arch in ['sinet', 'sbnet']:
        args.losses = ['xent', 'htri', 'infonce']

def main():
    # fix the seed in random operation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.log'))
    elif args.all_frames:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_eval_all_frames.log'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_eval_sampled_frames.log'))

    print_time("============ Initialized logger ============")
    print("\n".join("\t\t%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    print_time("============ Description ============")
    print_time("\t\t %s\n" % args.note)

    print_time("The experiment will be stored in %s\n" % args.save_dir)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed) 
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("Currently using CPU (GPU is highly recommended)")

    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        #单任务暂时不用
        consistency = 1
        consistency_rampup = 7.0
        return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)



    print_time("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)
    print_time("Initializing dataset {}".format(args.td))
    # t_dataset = data_manager.init_dataset(name=args.td, root="/HDD3/zxq/MARS/single/")
    # t_dataset = data_manager.init_dataset(name=args.td, root="/HDD3/zxq/")
    # t_dataset = data_manager.init_dataset(name=args.td)
    t_dataset = data_manager.init_dataset(name=args.td,root="/HDD3/zxq/CCVID")
    # t_dataset = data_manager.init_dataset(name=args.td,root="/HDD3/zxq/LS-VID/LS-VID")

    def get_data(l=1, shuffle=False):


        dataset = data_manager.init_dataset(name=args.td, root="/HDD3/zxq/")

        label_dict = {}
        for i, item_l in enumerate(dataset.train):
            if shuffle:
                labels = tuple([0 for i in range(l)])
                dataset.train[i] = (item_l[0],) + labels + (item_l[-1],)
            if item_l[1] in label_dict:
                label_dict[item_l[1]].append(i)
            else:
                label_dict[item_l[1]] = [i]

        return dataset, label_dict



    # Data augmentation
    spatial_transform_train = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.RandomHorizontalFlip(),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ST.RandomErasing()])

    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    temporal_transform_train = TT.TemporalRestrictedCrop(size=args.seq_len)
    temporal_transform_test = TT.TemporalRestrictedBeginCrop(size=args.seq_len)

    dataset_train = dataset.train
    dataset_query = dataset.query
    dataset_gallery = dataset.gallery

    pin_memory = True if use_gpu else False


    trainloader = DataLoader(
        VideoDataset1(
            dataset_train,
            spatial_transform=spatial_transform_train,
            temporal_transform=temporal_transform_train),
        # sampler=RandomIdentitySampler_vccvid(dataset.train, num_instances=args.num_instances),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True, )

    # t_trainloader = DataLoader(
    #     VideoDataset1(
    #         t_dataset.train,
    #         spatial_transform=spatial_transform_train,
    #         temporal_transform=temporal_transform_train),
    #     sampler=RandomIdentitySampler(t_dataset.train, num_instances=args.num_instances),
    #     # sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
    #     batch_size=args.train_batch, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=True, )

    t_trainloader = DataLoader(
        VideoDataset(
            t_dataset.train,
            spatial_transform=spatial_transform_train,
            temporal_transform=temporal_transform_train),
        # sampler=RandomIdentitySampler(t_dataset.train, num_instances=args.num_instances),
        sampler=RandomIdentitySampler_vccvid(t_dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True, )

    # def get_test_loader(batch_size=args.train_batch, workers=args.workers, testset=None):
    #
    #     if testset is None:
    #         testset = list(set(t_dataset.query) | set(t_dataset.gallery))
    #     # print(testset)
    #
    #     test_loader = DataLoader(
    #         Preprocessor_videoDataset1(testset, spatial_transform=spatial_transform_train,
    #         temporal_transform=temporal_transform_train),
    #         batch_size=batch_size, num_workers=workers,
    #         shuffle=False, pin_memory=True ,drop_last=True,)
    #
    #     return test_loader


    # queryloader_sampled_frames = DataLoader(
    #     VideoDataset1(dataset_query, spatial_transform=spatial_transform_test,
    #                  temporal_transform=temporal_transform_test),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False)
    #
    # galleryloader_sampled_frames = DataLoader(
    #     VideoDataset1(dataset_gallery, spatial_transform=spatial_transform_test,
    #                  temporal_transform=temporal_transform_test),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False)

    t_queryloader = DataLoader(
        VideoDataset(t_dataset.query, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False)

    t_galleryloader = DataLoader(
        VideoDataset(t_dataset.gallery, spatial_transform=spatial_transform_test,
                     temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False)

    # biaoji
    # queryloader_all_frames = DataLoader(
    #     VideoDatasetInfer(
    #         dataset_query, spatial_transform=spatial_transform_test, seq_len=args.seq_len),
    #     batch_size=1, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False)
    #
    # galleryloader_all_frames = DataLoader(
    #     VideoDatasetInfer(dataset_gallery, spatial_transform=spatial_transform_test, seq_len=args.seq_len),
    #     batch_size=1, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False)

    # t_queryloader_all_frames = DataLoader(
    #     VideoDatasetInfer1(
    #         dataset_query, spatial_transform=spatial_transform_test, seq_len=args.seq_len),
    #     batch_size=1, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False)
    #
    # t_galleryloader_all_frames = DataLoader(
    #     VideoDatasetInfer1(dataset_gallery, spatial_transform=spatial_transform_test, seq_len=args.seq_len),
    #     batch_size=1, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False)






    print_time("Initializing model: {}".format(args.arch))
    model = init_model(
                name=args.arch,
                num_classes = dataset.num_train_pids,
                losses=args.losses,
                seq_len=args.seq_len)
    #ema model
    ema_model = init_model(
                name=args.arch,
                num_classes = dataset.num_train_pids,
                losses=args.losses,
                seq_len=args.seq_len)




    attn_para_model = attn_para()
    # attn_para_model = attn_para00()

    # SSA_model  不需要opti在这里
    num_classes = dataset.num_train_pids  # 150 or 167
    # initialize SSA_model & trainer
    st_model = MultiStageModel(args, num_classes)
    trainer = Trainer(num_classes)

    print_time("Model Size w/o Classifier: {:.5f}M".format(
        sum(p.numel() for name, p in model.named_parameters() if 'classifier' not in name and 'projection' not in name)/1000000.0))

    criterions = {
        'xent': nn.CrossEntropyLoss(),
        'htri': TripletLoss(margin=args.margin, distance=args.distance),
        'infonce': InfoNce(num_instance=args.num_instances)}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    attn_para_optimizer = torch.optim.Adam(attn_para_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.pretrain:
        print("Loading pre-trained params from '{}'".format(args.pretrain_model_path))
        # pretrain_dict = torch.load(args.pretrain_model_path)
        pretrain_dict = torch.load(args.pretrain_model_path)['state_dict']
        model_dict = model.state_dict()
        state_dict_1 = {k: v for k, v in pretrain_dict.items() if (k != 'classifier.weight' and k!= 'classifier.bias')}
        model_dict.update(state_dict_1)
        model.load_state_dict(model_dict)

    if args.resume:

        # print(model)
        print_time("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # 改动fc层
        print(checkpoint['state_dict'].keys())
        state_dict_1 = {k: v for k, v in checkpoint['state_dict'].items() if (k != 'classifier.weight' and k!= 'classifier.bias')}
        print(state_dict_1.keys())

        model.load_state_dict(state_dict_1,strict=False)


        # print(model)


        # 注上解下
        # model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

        #ema model
        """
            创建一个EMA模型，该模型的参数与原始网络模型共享相同的数值，但在反向传播过程中不会更新EMA模型的参数。
            相反，EMA模型的参数会通过指数移动平均的方式根据原始网络模型的参数进行更新，以保持与原始模型的平滑一致性。
        """
        ema_model = nn.DataParallel(ema_model).cuda()
        for param in ema_model.parameters():
            param.detach_()


        attn_para_model.cuda()

    if args.evaluate:
        with torch.no_grad():
            if args.all_frames:
                print_time('==> Evaluate with [all] frames!')
                test(model, queryloader_all_frames, galleryloader_all_frames, use_gpu)
            else:
                print_time('==> Evaluate with sampled [{}] frames per video!'.format(args.seq_len))
                test(model, queryloader_sampled_frames, galleryloader_sampled_frames, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print_time("==> Start training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "/home/xiangqun/usr/temp_dir/models"
    results_dir = "/home/xiangqun/usr/temp_dir/results"

    #DCC loss and online clustering

    # class DCC(autograd.Function):
    #     @staticmethod
    #     def forward(ctx, inputs, targets, lut_ccc, lut_icc, momentum):
    #         ctx.lut_ccc = lut_ccc
    #         ctx.lut_icc = lut_icc
    #         ctx.momentum = momentum
    #         ctx.save_for_backward(inputs, targets)
    #         outputs_ccc = inputs.mm(ctx.lut_ccc.t())
    #         outputs_icc = inputs.mm(ctx.lut_icc.t())
    #
    #         return outputs_ccc, outputs_icc
    #
    #     @staticmethod
    #     def backward(ctx, grad_outputs_ccc, grad_outputs_icc):
    #         inputs, targets = ctx.saved_tensors
    #         grad_inputs = None
    #         if ctx.needs_input_grad[0]:
    #             grad_inputs = grad_outputs_ccc.mm(ctx.lut_ccc) + grad_outputs_icc.mm(ctx.lut_icc)
    #
    #         batch_centers = collections.defaultdict(list)
    #         for instance_feature, index in zip(inputs, targets.data.cpu().numpy()):
    #             batch_centers[index].append(instance_feature)
    #
    #         for y, features in batch_centers.items():
    #             mean_feature = torch.stack(batch_centers[y], dim=0)
    #             non_mean_feature = mean_feature.mean(0)
    #             x = F.normalize(non_mean_feature, dim=0)
    #             ctx.lut_ccc[y] = ctx.momentum * ctx.lut_ccc[y] + (1. - ctx.momentum) * x
    #             ctx.lut_ccc[y] /= ctx.lut_ccc[y].norm()
    #
    #         del batch_centers
    #
    #         for x, y in zip(inputs, targets.data.cpu().numpy()):
    #             ctx.lut_icc[y] = ctx.lut_icc[y] * ctx.momentum + (1 - ctx.momentum) * x
    #             ctx.lut_icc[y] /= ctx.lut_icc[y].norm()
    #
    #         return grad_inputs, None, None, None, None
    #
    # def oim(inputs, targets, lut_ccc, lut_icc, momentum=0.1):
    #     return DCC.apply(inputs, targets, lut_ccc, lut_icc, torch.Tensor([momentum]).to(inputs.device))
    #
    # import copy
    # class DCCLoss(nn.Module):
    #     def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
    #                  weight=None, size_average=True, init_feat=[]):
    #         super(DCCLoss, self).__init__()
    #         self.num_features = num_features
    #         self.num_classes = num_classes
    #         self.momentum = momentum
    #         self.scalar = scalar
    #         self.weight = weight
    #         self.size_average = size_average
    #
    #         self.register_buffer('lut_ccc', torch.zeros(num_classes, num_features).cuda())
    #         self.lut_ccc = copy.deepcopy(init_feat)
    #
    #         self.register_buffer('lut_icc', torch.zeros(num_classes, num_features).cuda())
    #         self.lut_icc = copy.deepcopy(init_feat)
    #
    #         print('Weight:{},Momentum:{}'.format(self.weight, self.momentum))
    #
    #     def forward(self, inputs, targets):
    #         inputs_ccc, inputs_icc = oim(inputs, targets, self.lut_ccc, self.lut_icc, momentum=self.momentum)
    #
    #         inputs_ccc *= self.scalar
    #         inputs_icc *= self.scalar
    #
    #         loss_ccc = F.cross_entropy(inputs_ccc, targets, size_average=self.size_average)
    #         loss_icc = F.cross_entropy(inputs_icc, targets, size_average=self.size_average)
    #
    #         loss_con = F.smooth_l1_loss(inputs_ccc, inputs_icc.detach(), reduction='elementwise_mean')
    #         loss = loss_ccc + loss_icc + loss_con
    #
    #         return loss
    #
    # def print_cluster_acc(label_dict, target_label_tmp):
    #     num_correct = 0
    #     for pid in label_dict:
    #         pid_index = np.asarray(label_dict[pid])
    #         pred_label = np.argmax(np.bincount(target_label_tmp[pid_index]))
    #         num_correct += (target_label_tmp[pid_index] == pred_label).astype(np.float32).sum()
    #     cluster_accuracy = num_correct / len(target_label_tmp)
    #     print(f'cluster accucary: {cluster_accuracy:.3f}')
    #
    # #end DCC
    #
    # # init cluster_loader
    # bzbz = 300
    # cluster_loader = get_test_loader(batch_size = bzbz,testset=sorted(t_dataset.train))
    #
    # # for batch_idx, (vids, pids, camid) in enumerate(cluster_loader):
    # #     print("Inputs shape:", vids.shape)
    # for i, (imgs, pids, _) in enumerate(cluster_loader):
    #     pseudo_labels=pids.numpy()
    #
    #
    # #
    # # print(len(np.where(pseudo_labels != -1)[0]))
    # # num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    # # del cluster_loader
    #
    # # print(num_cluster)
    # # print(pseudo_labels)
    # # print(pseudo_labels.shape)
    #
    # #kmeans！
    # # ncs = [40,80,120]
    # #
    # # prenc_i = -1
    # #
    # # target_features, _, _ = _feats_of_loader_nocc(
    # #     model,
    # #     cluster_loader,
    # #     feat_func=extract_feat_sampled_frames_whentrain,
    # #     use_gpu=use_gpu)
    # # print_time("Extracted features for model set, obtained {} matrix".format(target_features.shape))
    # #
    # # moving_avg_features = target_features.numpy()
    # # target_label = []
    # #
    # #
    # #
    # #
    # # for nc_i in ncs:
    # #     label_dict = {}
    # #     for i, item_l in enumerate(t_dataset.train):
    # #
    # #         labels = tuple([0 for i in range(nc_i)])
    # #         t_dataset.train[i] = (item_l[0],) + labels + (item_l[-1],)
    # #         if item_l[1] in label_dict:
    # #             label_dict[item_l[1]].append(i)
    # #         else:
    # #             label_dict[item_l[1]] = [i]
    # #
    # #
    # #     plabel_path = os.path.join(args.logs_dir, 'target_label{}_{}.npy'.format(nc_i, args.cluster_iter))
    # #     if os.path.exists(plabel_path):
    # #         target_label_tmp = np.load(plabel_path)
    # #         print('\n {} existing\n'.format(plabel_path))
    # #     else:
    # #         if prenc_i == nc_i:
    # #             target_label.append(target_label_tmp)
    # #             print_cluster_acc(label_dict, target_label_tmp)
    # #             continue
    # #
    # #         # km = KMeans(n_clusters=nc_i, random_state=args.seed, n_jobs=args.n_jobs).fit(moving_avg_features)
    # #         # target_label_tmp = np.asarray(km.labels_)
    # #         # cluster_centers = np.asarray(km.cluster_centers_)
    # #
    # #         cluster = faiss.Kmeans(2048, nc_i, niter=300, verbose=True, gpu=True)
    # #         cluster.train(moving_avg_features)
    # #         _, labels = cluster.index.search(moving_avg_features, 1)
    # #         target_label_tmp = labels.reshape(-1)
    # #
    # #     target_label.append(target_label_tmp)
    # #     print_cluster_acc(label_dict, target_label_tmp)
    # #     prenc_i = nc_i
    # # new_dataset = dataset_target.train
    #
    # def compute_group_labels(features, num_clusters):
    #     # 将特征转换为NumPy数组并连接起来
    #     features_combined = features.detach().cpu().numpy()
    #
    #     # 使用K-means算法进行聚类
    #     kmeans = KMeans(n_clusters=num_clusters)
    #     kmeans.fit(features_combined)
    #
    #     # 获取每个特征所属的簇索引
    #     labels = kmeans.labels_
    #
    #     # 将标签转换为PyTorch张量并放在CUDA上
    #     group_labels = torch.from_numpy(labels).cuda()
    #
    #     # 计算每个特征样本与其所在类簇中心点特征的距离
    #     cluster_centers = kmeans.cluster_centers_
    #     # distances = np.linalg.norm(features_combined - cluster_centers[labels], axis=1)
    #     # avg_distance = np.mean(distances)
    #
    #     # return group_labels, avg_distance
    #     return group_labels, cluster_centers
    #
    # def euclidean_distance(tensor1, tensor2):
    #     squared_diff = (tensor1 - tensor2) ** 2
    #     sum_squared_diff = torch.sum(squared_diff, dim=1)
    #     distance = torch.sqrt(sum_squared_diff)
    #     return distance
    #
    # class DistanceLoss(nn.Module):
    #     def __init__(self, cluster_features):
    #         super(DistanceLoss, self).__init__()
    #         self.cluster_features = cluster_features
    #
    #     def forward(self, inputs, pseudo_labels):
    #         batch_centers = self.cluster_features[pseudo_labels]
    #         loss = euclidean_distance(inputs, batch_centers)
    #         batch_loss = torch.mean(loss)  # 计算批量数据上的平均距离
    #         return batch_loss


    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        # train(epoch, model, criterions, optimizer, trainloader, use_gpu)

        # #clustering
        # with torch.no_grad():
        #     model.train()
        #     print('==> Create pseudo labels for unlabeled data')
        #     cluster_loader = get_test_loader(300, testset=sorted(t_dataset.train))
        #
        #
        #     features1, _, _ = _feats_of_loader_nocc(
        #         model,
        #         cluster_loader,
        #         feat_func=extract_feat_sampled_frames_whentrain,
        #         use_gpu=use_gpu)
        #     print_time("Extracted features for model set, obtained {} matrix".format(features1.shape))
        #
        #     features2, _, _ = _feats_of_loader_nocc(
        #         ema_model,
        #         cluster_loader,
        #         feat_func=extract_feat_sampled_frames_whentrain,
        #         use_gpu=use_gpu)
        #     print_time("Extracted features for ema_model set, obtained {} matrix".format(features2.shape))
        #
        #     # features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        #     # 将两个张量连接在一起
        #     features = torch.cat((features1, features2), dim=0)
        #     #
        #     #
        #     print(features.size())
        #     # rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)
        #     # print(rerank_dist.shape)
        #
        #
        #     # if epoch == 0:
        #     #     # DBSCAN cluster
        #     #     # eps = args.eps
        #     #     # eps = 0.6
        #     #     # print('Clustering criterion: eps: {:.3f}'.format(eps))
        #     #     # cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
        #     #     cluster = KMeans(n_clusters=40)
        #
        #
        #
        #
        #     # select & cluster images as training set of this epochs
        #     # pseudo_labels = cluster.fit_predict(rerank_dist)
        #     pseudo_labels, cluster_features = compute_group_labels(features,num_clusters=40)
        #     print("cluster_features.shape:", cluster_features.shape)
        #
        #     num_cluster = 40
        #     print(num_cluster)
        #     print(pseudo_labels)
        #
        #     # # generate new dataset and calculate cluster centers
        #     # @torch.no_grad()
        #     # def generate_cluster_features(labels, features):
        #     #     centers = collections.defaultdict(list)
        #     #     for i, label in enumerate(labels):
        #     #         if label == -1:
        #     #             continue
        #     #         centers[labels[i]].append(features[i])
        #     #
        #     #     centers = [
        #     #         torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        #     #     ]
        #     #
        #     #     centers = torch.stack(centers, dim=0)
        #     #     return centers
        #     #
        #     # cluster_features = generate_cluster_features(pseudo_labels, features).cuda()
        #     # print("cluster_features.shape:",cluster_features.shape)
        #     del cluster_loader, features
        #
        #
        #     pseudo_labeled_dataset1 = []
        #     for i, ((vids, _, cid), label) in enumerate(zip(sorted(t_dataset.train), pseudo_labels[:len(t_dataset.train)])):
        #         # if label != -1:
        #         pseudo_labeled_dataset1.append((vids, label.item(), cid))
        #
        #     pseudo_labeled_dataset2 = []
        #     for i, ((vids, _, cid), label) in enumerate(
        #             zip(sorted(t_dataset.train), pseudo_labels[len(t_dataset.train):])):
        #         # if label != -1:
        #         pseudo_labeled_dataset2.append((vids, label.item(), cid))
        #
        #     print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
        #
        #     new_t_train_loader1 = get_test_loader(batch_size=args.bS,testset=pseudo_labeled_dataset1)
        #     new_t_train_loader2 = get_test_loader(batch_size=args.bS,testset=pseudo_labeled_dataset2)
        #
        #
        # clu_loss = DistanceLoss(torch.from_numpy(cluster_features).cuda())
        # trainer.clu_loss = clu_loss

        # rank000 = test(model, queryloader_sampled_frames, galleryloader_sampled_frames, use_gpu)
        # rank000 = test1(model, queryloader_sampled_frames, galleryloader_sampled_frames, use_gpu)
        rank1 = test1(model, t_queryloader, t_galleryloader, use_gpu)

        trainer.train(ema_model,attn_para_model, attn_para_optimizer, st_model, model_dir, results_dir, device, args, epoch,
                      model, optimizer,criterions, trainloader, t_trainloader, use_gpu)
        # trainer.train(ema_model,attn_para_model, attn_para_optimizer, st_model, model_dir, results_dir, device, args, epoch,
        #               model, optimizer,criterions, trainloader, t_trainloader,new_t_train_loader1, new_t_train_loader2, use_gpu)
        train_time += round(time.time() - start_train_time)
        scheduler.step()

        if (epoch+1) >= args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print_time("==> Test")
            with torch.no_grad():
                model.eval()
                # rank000 = test1(model, queryloader_sampled_frames, galleryloader_sampled_frames, use_gpu)
                rank1 = test1(model, t_queryloader, t_galleryloader, use_gpu)


            is_best = rank1 > best_rank1
            if is_best: 
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
                attn_state_dict = attn_para_model.state_dict()
            else:
                state_dict = model.state_dict()
                attn_state_dict = attn_para_model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'attn_state_dict': attn_state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print_time("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print_time('=='*50)
    print_time("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

    # using all frames to evaluate the final performance after training
    args.all_frames = True

    infer_epochs = [150]
    if best_epoch !=150: infer_epochs.append(best_epoch)

    for epoch in infer_epochs:
        best_checkpoint_path = osp.join(args.save_dir, 'checkpoint_ep' + str(epoch) + '.pth.tar')
        checkpoint = torch.load(best_checkpoint_path)
        model.module.load_state_dict(checkpoint['state_dict'])

        print_time('==> Evaluate with all frames!')
        print_time("Loading checkpoint from '{}'".format(best_checkpoint_path))
        with torch.no_grad():
            test(model, queryloader_all_frames, galleryloader_all_frames, use_gpu)
        return


def _cal_dist(qf, gf, distance='cosine'):
    """
    :param logger:
    :param qf:  (query_num, feat_dim)
    :param gf:  (gallery_num, feat_dim)
    :param distance:
         cosine
    :return:
        distance matrix with shape, (query_num, gallery_num)
    """
    if distance == 'cosine':
        qf = F.normalize(qf, dim=1, p=2)
        gf = F.normalize(gf, dim=1, p=2)
        distmat = -torch.matmul(qf, gf.transpose(0, 1))
    else:
        raise NotImplementedError
    return distmat

def extract_feat_sampled_frames(model, vids, use_gpu=True):
    """
    :param model:
    :param vids: (b, 3, t, 256, 128)
    :param use_gpu:
    :return:
        features: (b, c)
    """
    if use_gpu: vids = vids.cuda()
    f = model(vids)    # (b, t, c)
    f = f.mean(-1)
    f = f.data.cpu()
    return f

def extract_feat_sampled_frames_whentrain(model, vids, use_gpu=True):
    """
    :param model:
    :param vids: (b, 3, t, 256, 128)
    :param use_gpu:
    :return:
        features: (b, c)
    """
    if use_gpu: vids = vids.cuda()
    _, _ ,_ ,f = model(vids)    # (b, t, c)
    f = f.mean(-1)
    f = f.data.cpu()
    return f

def extract_feat_all_frames(model, vids, max_clip_per_batch=45, use_gpu=True):
    """
    :param model:
    :param vids:    (_, b, c, t, h, w)
    :param max_clip_per_batch:
    :param use_gpu:
    :return:
        f, (1, C)
    """
    if use_gpu:
        vids = vids.cuda()
    _, b, c, t, h, w = vids.size()
    vids = vids.reshape(b, c, t, h, w)

    if max_clip_per_batch is not None and b > max_clip_per_batch:
        feat_set = []
        for i in range((b - 1) // max_clip_per_batch + 1):
            clip = vids[i * max_clip_per_batch: (i + 1) * max_clip_per_batch]
            f = model(clip)  # (max_clip_per_batch, t, c)
            f = f.mean(-1)
            feat_set.append(f)
        f = torch.cat(feat_set, dim=0)
    else:
        f = model(vids) # (b, t, c)
        f = f.mean(-1)   # (b, c)

    f = f.mean(0, keepdim=True)
    f = f.data.cpu()
    return f

def _feats_of_loader(model, loader, feat_func=extract_feat_sampled_frames, use_gpu=True):
    qf, q_pids, q_camids, q_clothes_ids = [], [], [], []

    pd = tqdm(total=len(loader), ncols=120, leave=False)
    for batch_idx, (vids, pids, camids, clothes_ids) in enumerate(loader):
        pd.update(1)

        f = feat_func(model, vids, use_gpu=use_gpu)
        qf.append(f)
        q_pids.extend(pids.numpy())
        q_camids.extend(camids.numpy())
        q_clothes_ids.extend(clothes_ids.numpy())

        # print(pids,camids,clothes_ids)
    pd.close()

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_clothes_ids = np.asarray(q_clothes_ids)

    return qf, q_pids, q_camids, q_clothes_ids

def _feats_of_loader_nocc(model, loader, feat_func=extract_feat_sampled_frames, use_gpu=True):
    qf, q_pids, q_camids = [], [], []

    pd = tqdm(total=len(loader), ncols=120, leave=False)
    for batch_idx, (vids, pids, camids) in enumerate(loader):
        pd.update(1)

        f = feat_func(model, vids, use_gpu=use_gpu)
        qf.append(f)
        q_pids.extend(pids.numpy())
        q_camids.extend(camids.numpy())
    pd.close()

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    return qf, q_pids, q_camids

def _eval_format_logger(cmc, mAP, ranks, desc=''):
    print_time("Results {}".format(desc))
    ptr = "mAP: {:.2%}".format(mAP)
    for r in ranks:
        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
    print_time(ptr)
    print_time("--------------------------------------")

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    model.eval()

    if args.all_frames:
        feat_func = extract_feat_all_frames
    else:
        feat_func = extract_feat_sampled_frames

    qf, q_pids, q_camids = _feats_of_loader_nocc(
        model,
        queryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = _feats_of_loader_nocc(
        model,
        galleryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    if args.dataset == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    time_elapsed = time.time() - since
    print_time('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print_time("Computing distance matrix")
    distmat = _cal_dist(qf=qf, gf=gf, distance=args.distance)
    print_time("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    _eval_format_logger(cmc, mAP, ranks, '')

    return cmc[0]

def test1(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    model.eval()

    if args.all_frames:
        feat_func = extract_feat_all_frames
    else:
        feat_func = extract_feat_sampled_frames

    qf, q_pids, q_camids, q_clothes_ids = _feats_of_loader(
        model,
        queryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids, g_clothes_ids = _feats_of_loader(
        model,
        galleryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    # print(q_camids)
    # print(g_camids)
    #
    # PATH1 = 'extract_features_rvccvid/query/'
    # PATH2 = 'extract_features_rvccvid/gallery/'
    # np.save('{}qf.npy'.format(PATH1),qf.cpu().numpy())
    # np.save('{}q_pids.npy'.format(PATH1),q_pids)
    # np.save('{}q_camids.npy'.format(PATH1),q_camids)
    # np.save('{}q_clothes_ids.npy'.format(PATH1),q_clothes_ids)
    #
    #
    # np.save('{}gf.npy'.format(PATH2), gf.cpu().numpy())
    # np.save('{}g_pids.npy'.format(PATH2), g_pids)
    # np.save('{}g_camids.npy'.format(PATH2), g_camids)
    # np.save('{}g_clothes_ids.npy'.format(PATH2), g_clothes_ids)




    time_elapsed = time.time() - since
    print_time('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print_time("Computing distance matrix")

    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i + 1], gf.t())).cpu()
    distmat = distmat.numpy()
    # q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    # g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    print('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    print("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='SC')
    print("Results ---------------------------------------------------")
    print(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")

    print("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='CC')
    print("Results ---------------------------------------------------")
    print(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")

    return cmc[0]


if __name__ == '__main__':
    specific_params(args)
    main()
