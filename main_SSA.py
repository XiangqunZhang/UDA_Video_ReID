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

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from models import init_model
from models.vidDA_model import attn_para, MultiStageModel
from utils.losses import TripletLoss, InfoNce
from utils.vidDA_train import Trainer
from utils.utils import AverageMeter, Logger, save_checkpoint, print_time
from utils.eval_metrics import evaluate,evaluate_with_clothes
from utils.samplers import RandomIdentitySampler,RandomIdentitySampler_vccvid
from utils import data_manager, ramps
from utils.video_loader import ImageDataset, VideoDataset, VideoDatasetcc

import torch.nn.functional as F
from torch import nn, autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bbbbb_ssssize = 32   #batch size
eeeepoch = 150 # epoch

parser = argparse.ArgumentParser(description='Train video model')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='v3dgait',
                    choices=data_manager.get_names())
parser.add_argument('--root', type=str, default='/HDD3/zxq')
parser.add_argument('--td', type=str, default='ccvid') #目标域数据集
parser.add_argument('--tdroot', type=str, default='/HDD3/zxq/') #目标域数据集地址
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

    print_time("Initializing source dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)
    print_time("Initializing target dataset {}".format(args.td))
    t_dataset = data_manager.init_dataset(name=args.td, root=args.tdroot)

    # t_dataset = data_manager.init_dataset(name=args.td, root="/HDD3/zxq/MARS/single/")
    # t_dataset = data_manager.init_dataset(name=args.td)
    # t_dataset = data_manager.init_dataset(name=args.td,root="/HDD3/zxq/CCVID")
    # t_dataset = data_manager.init_dataset(name=args.td,root="/HDD3/zxq/LS-VID/LS-VID")

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
    # dataset_query = dataset.query
    # dataset_gallery = dataset.gallery

    pin_memory = True if use_gpu else False

    def get_train_dataloader(dataset_name, dataset_train):
        if dataset_name in ['v3dgait','ccvid','rvccvid','svreid_cc', 'svreid', 'svreid_plus']: #cc type_reid
            dataset = VideoDatasetcc(dataset_train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train)
            sampler=RandomIdentitySampler_vccvid(dataset_train, num_instances=args.num_instances)
        else:
            dataset = VideoDataset(dataset_train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train)
            sampler=RandomIdentitySampler(dataset_train, num_instances=args.num_instances)

        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return dataloader

    def get_qg_dataloader(dataset_name, dataset_qg):
        if dataset_name in ['v3dgait','ccvid','rvccvid','svreid_cc', 'svreid', 'svreid_plus']:
            dataset = VideoDatasetcc(dataset_qg, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test)
        else:
            dataset = VideoDataset(dataset_qg, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test)


        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch,
            num_workers=args.workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False
        )

        return dataloader

    trainloader = get_train_dataloader(args.dataset, dataset_train)
    t_trainloader = get_train_dataloader(args.td, t_dataset.train)
    t_queryloader = get_qg_dataloader(args.td, t_dataset.query)
    t_galleryloader = get_qg_dataloader(args.td, t_dataset.gallery)

    print_time("Initializing video encoder")
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

    # SSA_model
    source_num_classes = dataset.num_train_pids  # 150 or 167
    target_num_classes = t_dataset.num_train_pids

    # initialize SSA_model & trainer
    st_model = MultiStageModel(args, source_num_classes)
    trainer = Trainer(source_num_classes)

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
        pretrain_dict = torch.load(args.pretrain_model_path)['state_dict']
        model_dict = model.state_dict()
        state_dict_1 = {k: v for k, v in pretrain_dict.items() if (k != 'classifier.weight' and k!= 'classifier.bias')}
        model_dict.update(state_dict_1)
        model.load_state_dict(model_dict)

    if args.resume:

        print_time("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # # fc层改动
        # state_dict_1 = {k: v for k, v in checkpoint['state_dict'].items() if (k != 'classifier.weight' and k!= 'classifier.bias')}
        # model.load_state_dict(state_dict_1,strict=False)

        model.load_state_dict(checkpoint['state_dict'])
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
            print_time('==> Evaluate with all frames!')
            if args.td in ['ccvid', 'rvccvid', 'svreid_cc', 'svreid', 'svreid_plus']:
                test_cc(model, t_queryloader, t_galleryloader, use_gpu)
            else:
                test(model, t_queryloader, t_galleryloader, use_gpu)

        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print_time("==> Start training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proj_path = os.getcwd()
    model_dir = proj_path + "/GRL_dir/models"
    results_dir = proj_path +"/GRL_dir/results"


    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()

        trainer.train(ema_model,attn_para_model, attn_para_optimizer, st_model, model_dir, results_dir, device, args, epoch,
                      model, optimizer,criterions, trainloader, t_trainloader, source_num_classes, target_num_classes, use_gpu)

        train_time += round(time.time() - start_train_time)
        scheduler.step()

        if (epoch+1) >= args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print_time("==> Test")
            with torch.no_grad():
                model.eval()
                if args.td in ['ccvid', 'rvccvid', 'svreid_cc', 'svreid', 'svreid_plus']:
                    rank1 = test_cc(model, t_queryloader, t_galleryloader, use_gpu)
                else:
                    rank1 = test(model, t_queryloader, t_galleryloader, use_gpu)

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
            if args.td in ['ccvid', 'rvccvid', 'svreid_cc', 'svreid', 'svreid_plus']:
                test_cc(model, t_queryloader, t_galleryloader, use_gpu)
            else:
                test(model, t_queryloader, t_galleryloader, use_gpu)
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

def test_cc(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
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