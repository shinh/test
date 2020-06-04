#!/usr/bin/env python3
#
# Usage:
#
# $ sample_imagenet.py ILSVRC2012/CLS-LOC imagenet_tiny
#

import argparse
import glob
import os
import random
import shutil


def sample_files(dirname, num):
    files = []
    for d in glob.glob(os.path.join(dirname, '*')):
        files.append(d)
    return random.sample(files, num)


def main():
    parser = argparse.ArgumentParser(description='Sample imagenet data')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('dest', metavar='DST', help='destination directory')
    parser.add_argument('--num-labels', default=8, type=int,
                        metavar='N', help='Number of target labels')
    parser.add_argument('--num-trains', default=256, type=int,
                        metavar='N', help='Number of training data per label')
    parser.add_argument('--num-vals', default=16, type=int,
                        metavar='N', help='Number of validation data per label')
    args = parser.parse_args()

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    assert os.path.exists(train_dir), train_dir
    assert os.path.exists(val_dir), val_dir

    labels = sample_files(train_dir, args.num_labels)
    labels = [os.path.basename(d) for d in labels]

    for kind, num in [('train', args.num_trains), ('val', args.num_vals)]:
        for label in labels:
            src_dir = os.path.join(args.data, kind, label)
            assert os.path.exists(src_dir), src_dir
            dst_dir = os.path.join(args.dest, kind, label)
            os.makedirs(dst_dir, exist_ok=True)

            for src in sample_files(src_dir, num):
                dst = os.path.join(dst_dir, os.path.basename(src))
                shutil.copy(src, dst)


if __name__ == '__main__':
    main()
