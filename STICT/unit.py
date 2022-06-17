import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.nn import functional as F

def get_frame_dict(image_path, image_ext='.png'):
    if 'MOS_U' in image_path:
        frame_list = sorted(glob(os.path.join(image_path, 'images/' + "*" + image_ext)))  
       
    else:
        with open(image_path + '/' + 'train.txt', 'r') as lines:
            samples = []
            for line in lines:
                samples.append(line.strip())
        frame_list = sorted(samples)
    if not frame_list:
        raise FileNotFoundError(image_path)
   
    frame_id_list = [f.split("/")[-1].replace(image_ext, "") for f in frame_list]    
    frame_dict = {}
    for frame_index, frame in enumerate(frame_id_list): 
        frame_dict[frame_index] = frame

    return frame_dict

def get_clips(frame_dict, clip_len=3, start_index=0):
  
    indexes = list(frame_dict.keys())
    inter = 1
    clips = []
    clip_start_index = start_index

    while clip_start_index < len(indexes) - clip_len:
        if frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 1]][:-10] \
                and frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 2]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index:clip_start_index + clip_len]})
            if frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 2]][:-10]:
                clip_start_index += inter
            else:
                clip_start_index += clip_len + start_index

        elif frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 1]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 1:clip_start_index + 2]})
            clip_start_index += inter  # - 1 + start_index
        else:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 2:clip_start_index + 1]})
            clip_start_index += inter  # - 2 + start_index
    if clip_start_index <= len(indexes):
        clips.append({'video_name': frame_dict[clip_start_index][:-10],
                      'clip_frame_index': indexes[len(indexes) - clip_len:len(indexes)]})
    return clips


def get_frame_dict_nv(image_path, image_ext='.png'):
   
    if 'MOS_U' in image_path:
        frame_list = sorted(glob(os.path.join(image_path, 'images/' + "*" + image_ext))) 
    else:
        with open(image_path + '/' + 'train.txt', 'r') as lines:
            samples = []
            for line in lines:
                samples.append(line.strip())
        frame_list = sorted(samples)

    if not frame_list:
        raise FileNotFoundError(image_path)
   
    frame_id_list = frame_list 
    frame_dict = {}
    for frame_index, frame in enumerate(frame_id_list): 
        frame_dict[frame_index] = frame

    return frame_dict


def get_clips_nv(frame_dict, clip_len=3, start_index=0):
    indexes = list(frame_dict.keys())

    inter = 1

    clips = []
    clip_start_index = start_index

    while clip_start_index < len(indexes) - clip_len:
        if frame_dict[indexes[clip_start_index]].split("/")[0] == frame_dict[indexes[clip_start_index + 1]].split("/")[
            0] \
                and frame_dict[indexes[clip_start_index]].split("/")[0] == \
                frame_dict[indexes[clip_start_index + 2]].split("/")[0]:
            clips.append({'video_name': frame_dict[clip_start_index],
                          'clip_frame_index': indexes[clip_start_index:clip_start_index + clip_len]})
            if frame_dict[indexes[clip_start_index]].split("/")[0] == \
                    frame_dict[indexes[clip_start_index + 2]].split("/")[0]:
                clip_start_index += inter
            else:
                clip_start_index += clip_len + start_index
    
        elif frame_dict[indexes[clip_start_index]].split("/")[0] == \
                frame_dict[indexes[clip_start_index + 1]].split("/")[0]:
            clips.append({'video_name': frame_dict[clip_start_index],
                          'clip_frame_index': indexes[clip_start_index - 1:clip_start_index + 2]})
            clip_start_index += inter  # - 1 + start_index
        else:
            clips.append({'video_name': frame_dict[clip_start_index],
                          'clip_frame_index': indexes[clip_start_index - 2:clip_start_index + 1]})
            clip_start_index += inter  # - 2 + start_index
    if clip_start_index <= len(indexes):
        clips.append({'video_name': frame_dict[clip_start_index],
                      'clip_frame_index': indexes[len(indexes) - clip_len:len(indexes)]})
   
    return clips


def get_frame_dict_test(image_path, image_ext='.png'):
    if 'MOS_U' in image_path:
        frame_list = sorted(glob(os.path.join(image_path, 'images/' + "*" + image_ext))) 
    else:
        with open(image_path + '/test.txt', 'r') as lines:
            samples = []
            for line in lines:
                samples.append(line.strip())
        frame_list = sorted(samples)

    if not frame_list:
        raise FileNotFoundError(image_path)
    frame_id_list = [f.split("/")[-1].replace(image_ext, "") for f in frame_list]  
    frame_dict = {}
    for frame_index, frame in enumerate(frame_id_list): 
        frame_dict[frame_index] = frame

    return frame_dict


def get_clips_test(frame_dict, clip_len=5, start_index=0):

    indexes = list(frame_dict.keys())

    inter = 5

    clips = []
    clip_start_index = start_index
   
    while clip_start_index < len(indexes) - clip_len:
        if frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 3]][:-10] \
                and frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 5]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index:clip_start_index + clip_len]})
            if frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 5]][:-10]:
                clip_start_index += inter
            else:
                clip_start_index += clip_len + start_index
       
        elif frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 4]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 1:clip_start_index + 4]})
            clip_start_index += inter - 1 + start_index
        elif frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 3]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 2:clip_start_index + 3]})
            clip_start_index += inter - 1 + start_index
        elif frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 2]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 3:clip_start_index + 2]})
            clip_start_index += inter - 2 + start_index
        elif frame_dict[indexes[clip_start_index]][:-10] == frame_dict[indexes[clip_start_index + 1]][:-10]:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 4:clip_start_index + 1]})
            clip_start_index += inter - 3 + start_index
        else:
            clips.append({'video_name': frame_dict[clip_start_index][:-10],
                          'clip_frame_index': indexes[clip_start_index - 5:clip_start_index]})
            clip_start_index += inter - 4 + start_index
     
    if clip_start_index <= len(indexes):
        clips.append({'video_name': frame_dict[clip_start_index][:-10],
                      'clip_frame_index': indexes[len(indexes) - clip_len:len(indexes)]})

    return clips


def readFlow(clip, frame_dict, flo_path):
    """ Read .flo file in Middlebury format"""
   
    name0 = frame_dict[clip['clip_frame_index'][0]]
    name1 = frame_dict[clip['clip_frame_index'][1]]
    name2 = frame_dict[clip['clip_frame_index'][2]]
    forward_flowfile = name0 + '-' + name1[-5:]
    backward_flowfile = name2 + '-' + name1[-5:]
    forward_flow_path = os.path.join(flo_path, 'forward', name0[:-6], forward_flowfile + '.flo')
    backward_flow_path = os.path.join(flo_path, 'backward', name0[:-6], backward_flowfile + '.flo')

    with open(forward_flow_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            forward_flow = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
    with open(backward_flow_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            backward_flow = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
    forward_flow = np.resize(forward_flow, (int(h), int(w), 2))
    backward_flow = np.resize(backward_flow, (int(h), int(w), 2))
    return forward_flow, backward_flow


class SuperResModel(nn.Module):
    class forwardWarp(nn.Module):
        def forward(self, img, flow, gridX, gridY):
            H, W = gridX.shape[1:]
            u = flow[:, 0, :, :]          
            v = flow[:, 1, :, :]
            x = gridX.expand_as(u).float() - u
            y = gridY.expand_as(v).float() - v
            x = 2 * (x / W - 0.5)
            y = 2 * (y / H - 0.5)
            grid = torch.stack((x, y), dim=3)
            imgOut = F.grid_sample(img, grid)
            return imgOut

    def __init__(self):
        super(SuperResModel, self).__init__()

        self.warper = self.forwardWarp()

    def forward(self, img, flow):
        gridX, gridY = np.meshgrid(np.arange(448), np.arange(448))
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        gridX = torch.tensor(gridX, requires_grad=False, device='cuda').repeat(num_devices, 1, 1)
        gridY = torch.tensor(gridY, requires_grad=False, device='cuda').repeat(num_devices, 1, 1)

        flow = flow.transpose(2, 0, 1)

        flow = torch.from_numpy(flow.astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        flow = flow.unsqueeze(0).to(device)
        img = img.unsqueeze(0)

        output = self.warper.forward(img, flow, gridX, gridY)

        return output

