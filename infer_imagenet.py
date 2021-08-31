# Infer imagenet with torchvision
#
# Usage:
#
# python3 infer_imagenet.py ~/datasets/imagenet/val mobilenetv3.mobilenet_v3_large --device cuda --bsize=50
#
# python3 infer_imagenet.py ~/datasets/imagenet/val quantized_mobilenetv3  --bsize=50
#

import argparse

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, device="cpu"):
    # batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(data_loader), [top1, top5], prefix="Test: ")

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            target = target.to(device)
            image = image.to(device)
            output = model(image)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if i % 50 == 0:
                progress.display(i)

            if i >= neval_batches:
                return top1, top5

    return top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="path to validation data"
    )
    parser.add_argument(
        "model", type=str, help="model name"
    )
    parser.add_argument(
        "--bsize", type=int, default=1, help="batch size"
    )
    parser.add_argument(
        "--device", default="cpu", help="device"
    )
    args = parser.parse_args()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.data,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.bsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    if args.model == "quantized_mobilenetv3":
        model =  torchvision.models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)
    else:
        m = torchvision.models
        for tok in args.model.split("."):
            m = getattr(m, tok)
        model = m(pretrained=True)

    print(evaluate(model, val_loader, 50000, args.device))

if __name__ == "__main__":
    main()
