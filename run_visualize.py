import os
import argparse
import numpy as np


def add_parser_arguments(parser):


    parser.add_argument("--gpu-id", type=int, default=0, help="which gpu to run on")
    parser.add_argument("--start", type=float, default=0, help='start value of delta')
    parser.add_argument("--end", type=float, default=0, help='end value of delta')
    parser.add_argument("--num-delta", type=int, default=0, help="number of delta to validate")
    parser.add_argument("--resume", type=str, default='checkpoints/resnet50-64-128-256-512-64/2020_07_11_08_12_17_009113/model_best.pth.tar', help='Which ckpt to visualize')
    parser.add_argument("--loss-surface-dir", type=str, default='./loss_surface/', help='where to save loss surface files')
    parser.add_argument("--widths", type=str, default='64-128-256-512-64', help='network arch')


def main():
    parser = argparse.ArgumentParser(description="visualize loss surface")

    add_parser_arguments(parser)
    args = parser.parse_args()
    print(args)

    if args.start > args.end or args.num_delta == 0:
        print("start < end or num-delta==0")
        exit()
    for seed in [0]:
        for delta in np.linspace(args.start, args.end, num=args.num_delta):

            cmd = 'CUDA_VISIBLE_DEVICES={} python main_prune_train.py --arch resnet --depth 20 --dataset cifar10 --evaluate --epochs 1 --resume {} --vis-seed={} --vis-delta={} --visualize --loss-surface-dir={}'.format(args.gpu_id, args.resume, seed, delta, args.loss_surface_dir, args.widths)

            print(cmd)

            os.system(cmd)
if __name__ == "__main__":
    main()
