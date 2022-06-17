#!/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import numpy as np
from misc import check_mkdir, cal_precision_recall_mae, AvgMeter, cal_fmeasure, cal_Jaccard, cal_BER
from torch.utils.data import DataLoader
import dataset
from net_STICT import SANet
import PIL.Image as Image
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--dataset_path', type=str, default='../data/DS/test/')
parser.add_argument('--dataset_txt_path', type=str, default='../data/DS/test/test.txt')
parser.add_argument('--trained_model', type=str, default='./DS')
parser.add_argument('--batch', '-b', default=1, type=int)


parser.set_defaults(bottleneck=True)
args = parser.parse_args()


def test(Network, Dataset):
    testdata = Dataset.TestDataset(data_path=args.dataset_path, txt_path=args.dataset_txt_path)
    testloader = DataLoader(testdata, batch_size=args.batch, shuffle=False, num_workers=8)

    net = Network()
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    net.cuda()

    precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()
    Jaccard_record = AvgMeter()
    BER_record = AvgMeter()
    shadow_BER_record = AvgMeter()
    non_shadow_BER_record = AvgMeter()

    with torch.no_grad():
        for image, mask, name_list in testloader:
            image = image.cuda().float()
            _, out2u, _, _, _, _, _, _ = net(image)
            out = out2u
            pred = (torch.sigmoid(out[:, 0, :, :]) * 255).cpu().numpy()
            mask = mask.cpu().numpy() * 255
            for (img, gt, name) in zip(pred, mask, name_list):

                if 'DS' in args.dataset_path:
                    device = np.array(Image.open(os.path.join(args.dataset_path, 'devices',  name+'.png')).convert('L'))
                    img = img * ((255 - device)//255)
                    gt = gt * ((255 - device)//255)
                img = img.astype(np.uint8)
                gt = gt.astype(np.uint8)
                precision, recall, mae = cal_precision_recall_mae(img, gt)
                Jaccard = cal_Jaccard(img, gt)
                Jaccard_record.update(Jaccard)
                BER, shadow_BER, non_shadow_BER = cal_BER(img, gt)
                BER_record.update(BER)
                shadow_BER_record.update(shadow_BER)
                non_shadow_BER_record.update(non_shadow_BER)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])
        log = 'MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(mae_record.avg, fmeasure,
                                                                                   Jaccard_record.avg, BER_record.avg,
                                                                                   shadow_BER_record.avg,
                                                                                   non_shadow_BER_record.avg)
        print(log)

    return log


if __name__ == '__main__':
    test(SANet, dataset)
