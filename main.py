from utils import get_config  
import argparse
from tools.train import train
from tools.test import test
import numpy as np
import torch

default_config_path = 'configs/vit.yaml'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    cfg = get_config(default_config_path)
    args =  parse_args()
    if hasattr(args, 'config') and args.config:
        cfg = get_config(args.config)
    else:
        cfg = get_config(default_config_path)

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    

    if hasattr(args, 'output_dir') and args.output_dir:
        cfg['output_dir'] = args.output_dir

    if args.mode == 'train':
        train(cfg)
    elif args.mode == 'test':
        test(cfg)

