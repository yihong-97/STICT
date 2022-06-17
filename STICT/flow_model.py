import torch.nn as nn
from torch.nn import functional as F
import torch
import os
import sys
import functools
from flownet import *
from flownet.resample2d_package.resample2d import Resample2d

affine_par = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
BatchNorm2d = functools.partial(nn.BatchNorm2d)

TAG_CHAR = np.array([202021.25], np.float32)


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)  
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


class FlowModel(nn.Module):
    def __init__(self, student, criterion, args):
        super(FlowModel, self).__init__()
        self.flownet = FlowNet2(args, requires_grad=False)
        checkpoint = torch.load(
            "./pretrained_model/FlowNet2_checkpoint.pth.tar")
        self.flownet.load_state_dict(checkpoint['state_dict'])
        self.flow_warp = Resample2d()
    
    
    def warp(self,  s_image, s_frames):
        with torch.no_grad():
            flow = self.flownet(s_image, s_frames)
      

        warp = self.flow_warp(s_frames, flow)
        
        return warp


