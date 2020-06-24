import glob
import os
import sys


train_dir = sys.argv[1]

labels = []
for dir in sorted(glob.glob(os.path.join(train_dir, '*'))):
    labels.append(os.path.basename(dir))

for label_id, label in enumerate(labels):
    for fn in sorted(glob.glob(os.path.join(train_dir, label, '*'))):
        print(fn, label_id)
