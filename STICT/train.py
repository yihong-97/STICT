#!/usr/bin/python3
# coding=utf-8
import sys
import os
import datetime

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import random
import ramps
import argparse
import warnings

torch.cuda.manual_seed_all(2021)
torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)

import config
import unit
import dataset
import dataset_u
from net_STICT import SANet

import functools
from flownet.FlowNet2 import *
from flownet.resample2d_package.resample2d import Resample2d

affine_par = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
BatchNorm2d = functools.partial(nn.BatchNorm2d)
TAG_CHAR = np.array([202021.25], np.float32)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')

parser.add_argument('--consistency', type=float, default=2.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=5.0, help='consistency_rampup')

parser.add_argument('--batch_label', '-bl', default=4, type=int)
parser.add_argument('--batch_unlabel', '-ubl', default=4, type=int)
parser.add_argument('--epochs', '-e', default=64, type=int)
parser.add_argument('--lr', '-lr', default=0.03, type=float)
parser.add_argument('--target_domain', type=str, default='DS_U', help='DS_U, MOS_U, ViSha')

parser.add_argument('--dataset_path', type=str, default='./data/SBU-shadow/SBUTrain4KRecoveredSmall')
parser.add_argument('--dataset_txt_path', type=str, default='./data/SBU-shadow/SBUTrain4KRecoveredSmall/train.txt')
parser.add_argument('--dataset_U_path', type=str, default='./data/DS/train/')

parser.set_defaults(bottleneck=True)
args = parser.parse_args()


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sigmoid_mse_loss(input_logits, target_logits):
    """Takes sigmoid on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = F.sigmoid(input_logits)
    target_sigmoid = F.sigmoid(target_logits)

    return F.mse_loss(input_sigmoid, target_sigmoid, size_average=True)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--config', type=str, default='config/flownet/flow.yaml',
                        help='config file')
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def train(Dataset, Dataset_U, Network):
    logs = {'lr': [], 'trainloss': [], 'BER': []}
    ## source image dataset
    data = Dataset.Data(data_path=args.dataset_path,
                        txt_path=args.dataset_txt_path)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=args.batch_label, shuffle=True, num_workers=0)

    ## target video dataset
    if args.target_domain == 'DS_U':
        frame_dict = unit.get_frame_dict(args.dataset_U_path)
        clips = unit.get_clips(frame_dict)
    else:
        frame_dict = unit.get_frame_dict_nv(args.dataset_U_path)
        clips = unit.get_clips_nv(frame_dict)
    data_u = Dataset_U.Data(data_path=args.dataset_U_path, frame_dict=frame_dict, clips=clips)
    loader_u = DataLoader(data_u, batch_size=args.batch_unlabel, shuffle=True, num_workers=0)

    net = Network()
    net.train(True)
    net.cuda()

    net_ema = Network()
    for param in net_ema.parameters():
        param.detach_()
    net_ema.train(True)
    net_ema.cuda()

    args_flo = get_parser()
    args_flo.rgb_max = 1.0
    args_flo.fp16 = False
    flownet = FlowNet2(args_flo, requires_grad=False)
    flownet.cuda()
    checkpoint = torch.load("./pretrained_model/FlowNet2_checkpoint.pth.tar")
    flownet.load_state_dict(checkpoint['state_dict'])
    flow_warp = Resample2d()

    ## parameter
    print('-------------Train Setting-----------------\n')
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.Adam([{'params': base}, {'params': head}], lr=args.lr)

    global_step = 0
    co_w = 0.0
    for epoch in range(args.epochs):
        optimizer.param_groups[0]['lr'] = (1 - ((epoch) / (args.epochs))) * args.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - ((epoch) / (args.epochs))) * args.lr

        logs['lr'].append(optimizer.param_groups[1]['lr'])
        net.train(True)

        step = 0
        loss_mean = 0
        for (image, mask), (image_u0, image_u1, image_u2, image_u0f, image_u1f, image_u2f, clips) in zip((loader), (
                loader_u)):
            image, mask = image.cuda().float(), mask.cuda().float()
            out1u_l, out2u_l, out2r_l, out3r_l, out4r_l, out2t_l, out3t_l, out4t_l = net(image)

            ## loss_sup
            loss1u = structure_loss(out1u_l, mask)
            loss2u = structure_loss(out2u_l, mask)

            loss2r = structure_loss(out2r_l, mask)
            loss3r = structure_loss(out3r_l, mask)
            loss4r = structure_loss(out4r_l, mask)

            loss2t = structure_loss(out2t_l, mask)
            loss3t = structure_loss(out3t_l, mask)
            loss4t = structure_loss(out4t_l, mask)

            loss_sl = (
                                  loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 + loss4r / 8 + loss2t / 2 + loss3t / 4 + loss4t / 8

            ##
            image_u0 = image_u0.cuda().float()
            image_u1 = image_u1.cuda().float()
            image_u2 = image_u2.cuda().float()
            noise0 = torch.clamp(torch.randn_like(image_u0) * 0.1, -0.2, 0.2)
            ema_inputs0 = image_u0 + noise0
            ema_inputs1 = image_u1 + noise0
            ema_inputs2 = image_u2 + noise0

            image_u0f = image_u0f.cuda().float()
            image_u1f = image_u1f.cuda().float()
            image_u2f = image_u2f.cuda().float()

            rate_W, rate_H = image_u1.shape[2:]
            rate = torch.rand(1, 1, int(rate_W / 32), int(rate_H / 32)).cuda()

            out1u_u1, out2u_u1, out2u_u1Fa_, out2u_u1mix, out2r_u1, out3r_u1, out4r_u1, out2t_u1, out3t_u1, out4t_u1 = net(
                image_u1, rate)

            with torch.no_grad():
                out1u_u0, out2u_u0, out2u_u0Fa_, out2u_u0mix, out2r_u0, out3r_u0, out4r_u0, out2t_u0, out3t_u0, out4t_u0 = net(
                    image_u0, rate)
                out1u_u2, out2u_u2, out2u_u2Fa_, out2u_u2mix, out2r_u2, out3r_u2, out4r_u2, out2t_u2, out3t_u2, out4t_u2 = net(
                    image_u2, rate)

            with torch.no_grad():
                out1u_eb0, out2u_eb0, out2u_eb0Fa_, out2u_eb0mix, out2r_eb0, out3r_eb0, out4r_eb0, out2t_eb0, out3t_eb0, out4t_eb0 = net_ema(
                    ema_inputs0, rate)
                out1u_eb1, out2u_eb1, out2u_eb1Fa_, out2u_eb1mix, out2r_eb1, out3r_eb1, out4r_eb1, out2t_eb1, out3t_eb1, out4t_eb1 = net_ema(
                    ema_inputs1, rate)
                out1u_eb2, out2u_eb2, out2u_eb2Fa_, out2u_eb2mix, out2r_eb2, out3r_eb2, out4r_eb2, out2t_eb2, out3t_eb2, out4t_eb2 = net_ema(
                    ema_inputs2, rate)

                rate = F.interpolate(rate, size=(rate_W, rate_H), mode='bilinear')

            ## loss_sic
            mixresult0 = rate * out2u_eb0 + (1 - rate) * out2u_eb0Fa_
            mixresult1 = rate * out2u_eb1 + (1 - rate) * out2u_eb1Fa_
            mixresult2 = rate * out2u_eb2 + (1 - rate) * out2u_eb2Fa_

            loss_mix0 = sigmoid_mse_loss(mixresult0, out2u_u0mix)
            loss_mix1 = sigmoid_mse_loss(mixresult1, out2u_u1mix)
            loss_mix2 = sigmoid_mse_loss(mixresult2, out2u_u2mix)
            loss_sic = loss_mix0 + loss_mix1 + loss_mix2

            ## loss_sc
            label = (
                            out1u_eb1 + out2u_eb1 + out2r_eb1 + out3r_eb1 + out4r_eb1 + out2t_eb1 + out3t_eb1 + out4t_eb1) / 8.0

            loss1u_s = sigmoid_mse_loss(out1u_u1, label)
            loss2u_s = sigmoid_mse_loss(out2u_u1, label)

            loss2r_s = sigmoid_mse_loss(out2r_u1, label)
            loss3r_s = sigmoid_mse_loss(out3r_u1, label)
            loss4r_s = sigmoid_mse_loss(out4r_u1, label)

            loss2t_s = sigmoid_mse_loss(out2t_u1, label)
            loss3t_s = sigmoid_mse_loss(out3t_u1, label)
            loss4t_s = sigmoid_mse_loss(out4t_u1, label)

            loss_scl = (loss1u_s + loss2u_s + loss2r_s + loss3r_s + loss4r_s + loss2t_s + loss3t_s + loss4t_s) / 8

            ## warp
            with torch.no_grad():
                image_forward_flow = flownet(image_u1f, image_u0f)
                image_backward_flow = flownet(image_u1f, image_u2f)

            scale_pred0_t = torch.sigmoid(out2u_eb0)
            scale_pred1_t = torch.sigmoid(out2u_eb1)
            scale_pred2_t = torch.sigmoid(out2u_eb2)

            scale_pred0 = torch.sigmoid(out2u_u0)
            scale_pred1 = torch.sigmoid(out2u_u1)
            scale_pred2 = torch.sigmoid(out2u_u2)

            warp_i_forward = flow_warp(image_u0, image_forward_flow)
            warp_i_backward = flow_warp(image_u2, image_backward_flow)

            noc_mask_forward = torch.exp(-1 * torch.abs(torch.sum(image_u1 - warp_i_forward, dim=1))).unsqueeze(1)
            noc_mask_backward = torch.exp(-1 * torch.abs(torch.sum(image_u1 - warp_i_backward, dim=1))).unsqueeze(1)

            warp_o_forward_t = flow_warp(scale_pred0_t, image_forward_flow)
            warp_o_backward_t = flow_warp(scale_pred2_t, image_backward_flow)

            criterion_flow = nn.MSELoss(size_average=True)

            ## loss_tic
            loss_tic_f = criterion_flow(scale_pred1 * noc_mask_forward, warp_o_forward_t * noc_mask_forward)
            loss_tic_b = criterion_flow(scale_pred1 * noc_mask_backward, warp_o_backward_t * noc_mask_backward)
            loss_tic = loss_tic_f + loss_tic_b

            ## loss consistency
            loss_co = 0.1 * (loss_scl + loss_tic + loss_sic)

            ## total loss
            co_w = get_current_consistency_weight(epoch)
            loss = loss_sl + co_w * loss_co

            loss_mean += loss_sl.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(net, net_ema, args.ema_decay, global_step)

            global_step += 1
            step += 1
        print('%s | step:%d/%d | weight=%.6f | lr=%.6f | loss=%.6f ' % (
            datetime.datetime.now(), epoch + 1, args.epochs, co_w, optimizer.param_groups[1]['lr'], loss_mean))
        torch.save(net.state_dict(), './out/model-' + str(epoch + 1))


if __name__ == '__main__':
    train(dataset, dataset_u, SANet)
