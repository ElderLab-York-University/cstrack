#!/usr/bin/env python
# -*- coding: utf-8 -*-

r'''Script for generating a TensorRT engine file for CSTrack.

    Author: Helio Perroni Filho
'''


import _init_paths

import argparse
import os
from subprocess import run

import torch
from torch.onnx import export

from tracker.cstrack import JDETracker


def main():
    r'''Convert the CSTrack model to ONNX and then TensorRT.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--nms_thres', type=float, default=0.6, help='iou thresh for nms')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument('--mean', type=float, default=[0.408, 0.447, 0.470], help='mean for STrack')
    parser.add_argument('--std', type=float, default=[0.289, 0.274, 0.278], help='std for STrack')

    parser.add_argument('--weights',type=str, default='cfg/cstrack.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='cfg/model_set/CSTrack.yaml', help='model.yaml path')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')

    opt = parser.parse_args()
    opt.device = -1 # Device must be set to CPU during ONNX file generation.
    opt.img_size = (960, 960)
    opt.gpus = [opt.device]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)

    dummy_input = torch.randn(1, 3, *opt.img_size[::-1], device=('cuda' if opt.gpus[0] >= 0 else 'cpu'))

    tracker = JDETracker(opt)
    model = tracker.model

    export(model, dummy_input, 'cfg/cstrack.onnx', opset_version=11)

    # Set device to GPU for TensorRT file generation.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    run(['/usr/src/tensorrt/bin/trtexec', '--useCudaGraph', '--onnx=cfg/cstrack.onnx', '--saveEngine=cfg/cstrack.trt'])


if __name__ == '__main__':
    main()
