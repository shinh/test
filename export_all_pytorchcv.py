import argparse
import os
import subprocess

from pytorchcv import model_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsize', type=int, default=1)
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    for model in dir(model_provider):
        if not model[0].islower():
            continue
        print("Export %s..." % model)
        subprocess.call([
            "python3",
            "export_pytorchcv.py",
            model,
            "--bsize=%d" % args.bsize,
            "--size=%d" % args.size,
        ])


if __name__ == '__main__':
    main()
