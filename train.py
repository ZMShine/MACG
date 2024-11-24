from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from line_profiler import profile

#指定计算显卡序号
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')

import argparse
import yaml
from model.Supervisor import CGLMSupervisor

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor = CGLMSupervisor(temperature=args.temperature, **supervisor_config)
        supervisor.train()


if __name__ == '__main__':
    num_experiments = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", type=str, default="./data/Para/para_bay.yaml",
                        help="Configuration filename for restoring the model.")
    parser.add_argument("--use_cpu_only", default=False, type=bool, help="Set to true to only use cpu.")
    parser.add_argument("--temperature", default=0.9, type=float, help="temperature value for gumbel-softmax.")

    for _ in range(num_experiments):
        args = parser.parse_args()
        main(args)