import glob
import os
import random
import sys


random.seed(42)

train_dir = sys.argv[1]

labels = []
for dir in sorted(glob.glob(os.path.join(train_dir, '*'))):
    labels.append(os.path.basename(dir))

data = []
for label_id, label in enumerate(labels):
    for fn in sorted(glob.glob(os.path.join(train_dir, label, '*'))):
        data.append((fn, label_id))

for i in range(10):
    random.shuffle(data)
    for fn, id in data:
        print(fn, id)
