import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from medpy import metric
import argparse


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_precision_recall_mae(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    prediction_bool = (prediction > 0.5)
    gt_bool = (gt > 0.5)
    prediction_bool = prediction_bool.astype(np.float)
    gt_bool = gt_bool.astype(np.float)

    mae = np.mean(np.abs(prediction_bool - gt_bool))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae

def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure

def cal_Jaccard(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    prediction = prediction / 255.
    gt = gt / 255.

    pred = (prediction > 0.5)
    gt = (gt > 0.5)
    Jaccard = metric.binary.jc(pred, gt)

    return Jaccard

def cal_BER(prediction, label, thr = 128):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    BER = 0.5 * (2 - TP / Np - TN / Nn) * 100
    shadow_BER = (1 - TP / Np) * 100
    non_shadow_BER = (1 - TN / Nn) * 100
   
    return BER, shadow_BER, non_shadow_BER

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-gp', '--gt_path', type=str, default='./data/DS/test/')
    parser.add_argument('-pp', '--pred_path', type=str, default='./result/MTMT-SSL/DS/')
    parser.set_defaults(bottleneck=True)
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path 

    print('evalute the predictions: ', pred_path)

    precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()
    Jaccard_record = AvgMeter()
    BER_record = AvgMeter()
    shadow_BER_record = AvgMeter()
    non_shadow_BER_record = AvgMeter()

    video_list = os.listdir(pred_path)

    for video in tqdm(video_list):
        gt_list = os.listdir(os.path.join(gt_path, 'labels', video))    
        img_list = [f for f in os.listdir(os.path.join(pred_path, video))]  
        img_set = list(set([img.split('/')[-1] for img in img_list]))  
        for img_prefix in img_set:

            gt = np.array(Image.open(os.path.join(gt_path, 'labels', video, img_prefix)).convert('L'))

            width, height = gt.shape
            img = np.array(Image.open(os.path.join(pred_path, video, img_prefix)).convert('L').resize((height, width)))

            if 'DS' in gt_path:
                device = np.array(Image.open(os.path.join(gt_path, 'devices', video, img_prefix)).convert('L'))
                img = img * ((255 - device)//255)
                gt = gt * ((255 - device)//255)

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
    log = 'MAE:{:.3f}, F-beta:{:.3f}, Jaccard:{:.3f}, BER:{:.2f}, SBER:{:.2f}, non-SBER:{:.2f}'.format(mae_record.avg, fmeasure, Jaccard_record.avg, BER_record.avg, shadow_BER_record.avg, non_shadow_BER_record.avg)
    print(log)


