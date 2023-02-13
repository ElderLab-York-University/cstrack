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

from tracker.cstrack import CSTrack


def main():
    r'''Convert the CSTrack model to ONNX and then TensorRT.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights',type=str, default='cfg/cstrack.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='cfg/model_set/CSTrack.yaml', help='model.yaml path')
    parser.add_argument('--skip_trt', action='store_true', help='whether to skip exporting the ONNX model to TensorRT')

    opt = parser.parse_args()
    opt.img_size = (960, 960)

    # Device must be set to CPU during ONNX file generation.
    opt.device = -1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
    opt.gpus = [opt.device]

    dummy_input = torch.randn(1, 3, *opt.img_size[::-1], device='cpu')

    tracker = CSTrack(opt)
    model = tracker.model

    export(model, dummy_input, 'cstrack.onnx', opset_version=11)

    run(
        'python -m onnxoptimizer cstrack.onnx cfg/cstrack.onnx --passes $(python -m onnxoptimizer --print_all_passes)',
        shell=True, check=True
    )

    os.remove('cstrack.onnx')

    if opt.skip_trt:
        return

    # Set device to GPU for TensorRT file generation.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    run([
        '/usr/src/tensorrt/bin/trtexec',
        '--fp16', '--useCudaGraph', '--useSpinWait',
        '--onnx=cfg/cstrack.onnx',
        '--saveEngine=cfg/cstrack.trt'
    ])


if __name__ == '__main__':
    main()
