import argparse

import numpy as np
import torch
import torch.onnx.symbolic_helper
import torch.onnx.symbolic_registry
import torchvision


OPSET_VERSION = 13


def register_hswish(opset):
    @torch.onnx.symbolic_helper.parse_args("v")
    def symbolic_hardswish(g, input):
        return g.op("HardSwish", input)

    torch.onnx.symbolic_registry.register_op(
        "hardswish", symbolic_hardswish, "", opset
    )


def create_random_input(bsize, h=224, w=224):
    x = np.random.randint(0, 256, (bsize, 3, h, w)).astype(np.float32)
    return torch.tensor(x)


def main():
    parser = argparse.ArgumentParser(description="Export mobilenetv3")
    parser.add_argument("model")
    parser.add_argument("--bsize", type=int, default=1)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--hswish", action="store_true")
    args = parser.parse_args()

    bsize = args.bsize
    models = dir(torchvision.models) if args.model == "all" else [args.model]

    if args.hswish:
        register_hswish(args.opset)

    x = create_random_input(bsize)
    for model_name in models:
        model_fn = getattr(torchvision.models, model_name)
        if type(model_fn) != type(main):
            continue

        onnx_name = f'{model_name}_bs{bsize}.onnx'
        print('Exporting ' + onnx_name)
        torch.onnx.export(model_fn(pretrained=True),
                          x, onnx_name,
                          opset_version=args.opset,
                          input_names=["input"])


if __name__ == '__main__':
    main()
