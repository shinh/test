import numpy as np
import torch
import torchvision


def create_random_input(bsize, h=224, w=224):
    x = np.random.randint(0, 256, (bsize, 3, h, w)).astype(np.float32)
    return torch.tensor(x)


def main():
    for bsize in [1, 128]:
        x = create_random_input(bsize)

        for model_name in dir(torchvision.models):
            model_fn = getattr(torchvision.models, model_name)
            if type(model_fn) != type(main):
                continue

            onnx_name = f'{model_name}_bs{bsize}.onnx'
            print('Exporting ' + onnx_name)
            torch.onnx.export(model_fn(), x, onnx_name)


if __name__ == '__main__':
    main()
