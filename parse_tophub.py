#!/usr/bin/python3
#
# Usage:
#
# parse_tophub.py ~/.tvm/tophub/cuda_v0.04.log

import json
import sys


with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith('#'):
            continue
        obj = json.loads(line)
        i = obj['i']
        target = i[0]
        func = i[1]
        bs = i[2][0][1][0]
        print('%s %s bs=%s' % (target, func, bs))


