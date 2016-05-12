#!/usr/bin/python

import sys

class Node(object):
  def __init__(self):
    self.rule = None
    self.inputs = []
    self.outputs = []

if len(sys.argv) < 2:
  print('Usage: %s build.ninja [from] [to]' % sys.argv[0])
  sys.exit(1)

build_ninja = sys.argv[1]
from_target = sys.argv[2] if len(sys.argv) >= 3 else None
to_target = sys.argv[3] if len(sys.argv) >= 4 else None

nodes = {}
defaults = None
sys.stderr.write('Parsing %s...\n' % build_ninja)
with open(build_ninja) as f:
  for line in f:
    line = line.rstrip()
    if line.startswith('build '):
      toks = line.split(' ')
      output = toks[1][0:-1]
      rule = toks[2]
      inputs = toks[3:]

      node = Node()
      node.rule = rule
      node.inputs = inputs
      nodes[output] = node

      depfile = None
      is_restat = False
    elif line.startswith('default '):
      assert not defaults
      defaults = line.split(' ')[1:]
    elif line.startswith(' depfile = '):
      assert not depfile
      depfile = line.split(' ')[3]
    elif line.startswith(' restat = 1'):
      is_restat = True
      pass

assert defaults

sys.stderr.write('Create reverse edges...\n')
for output, node in nodes.iteritems():
  for i in node.inputs:
    if i in nodes:
      nodes[i].outputs.append(output)

if to_target is None:
  candidates = nodes.keys()
else:
  candidates = set()
  q = [to_target]
  while q:
    name = q.pop()
    if name in candidates:
      continue
    candidates.add(name)

    for next_node in nodes[name].outputs:
      q.append(next_node)

if from_target:
  q = [(from_target, 0)]
else:
  q = [(n, 0) for n in defaults]

while q:
  name, depth = q.pop()
  if name not in candidates:
    continue
  print ' ' * depth + name

  if nodes[name].inputs:
    for next_node in reversed(nodes[name].inputs):
      q.append((next_node, depth+1))
