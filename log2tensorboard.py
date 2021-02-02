# Usage:
#
# $ log2tb.py log out_dir
# $ tensorboard --log_dir out_dir
#

import json
import sys

import torch
from torch.utils import tensorboard

with open(sys.argv[1]) as f:
    records = json.load(f)

writer = tensorboard.SummaryWriter(sys.argv[2])

for record in records:
    iteration = None
    for key, value in record.items():
        if key == 'iteration':
            iteration = value

    assert iteration is not None

    for key, value in record.items():
        if not isinstance(value, (int, float)):
            continue
        writer.add_scalar(key, value, iteration)

writer.close()
