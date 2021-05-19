import argparse

import torch

from pytorchcv import model_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--bsize', type=int, default=1)
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    model = getattr(model_provider, args.model)()
    x = torch.zeros(args.bsize, 3, args.size, args.size)
    name = '%s_bsize%d_%dx%d' % (args.model, args.bsize, args.size, args.size)
    torch.onnx.export(
        model, x, name + '.onnx',
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == '__main__':
    main()
