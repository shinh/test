import sys
from z3 import *


def parse(s):
    return s.strip().splitlines()


def step(w):
    H = len(w)
    W = len(w[0])
    nw = []
    for y in range(H):
        nr = ""
        for x in range(W):
            c = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny = (y + dy + H) % H
                    nx = (x + dx + W) % W
                    c += int(w[ny][nx] == "#")
            if c == 3 or (c == 4 and w[y][x] == "#"):
                nr += "#"
            else:
                nr += "."
        nw.append(nr)
    return nw


def z3_step(w, solver):
    H = len(w)
    W = len(w[0])
    nw = []
    for y in range(H):
        nr = []
        for x in range(W):
            cs = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny = (y + dy + H) % H
                    nx = (x + dx + W) % W
                    cs.append(w[ny][nx])
            c = sum(cs)
            e = Or(c == 3, And(c == 4, w[y][x]))
            # solver.add(c < 6)
            nr.append(e)
        nw.append(nr)
    return nw


def z3_step_n(w, nsteps, solver):
    for _ in range(nsteps):
        w = z3_step(w, solver)
    return w


def rev_step(w, nsteps):
    H = len(w)
    W = len(w[0])
    solver = Solver()

    cells = []
    for y in range(H):
        row = []
        for x in range(W):
            row.append(Bool(f"cell_{x}_{y}"))
        cells.append(row)

    ncells = z3_step_n(cells, nsteps, solver)
    for y in range(H):
        for x in range(W):
            e = ncells[y][x]
            if w[y][x] == "#":
                solver.add(e)
            else:
                solver.add(Not(e))

    res = solver.check()
    assert res == sat

    m = solver.model()
    nw = []
    for y in range(H):
        nr = ""
        for x in range(W):
            if m[cells[y][x]]:
                nr += "#"
            else:
                nr += "."
        nw.append(nr)
    return nw


glider = parse(r"""
........
........
...##...
...#.#..
...#....
""")


def test(w):
    for i in range(10):
        print(f"=== {i} ===")
        print("\n".join(w))
        w = step(w)

# test(glider)

test(parse(open(sys.argv[1]).read()))


def test_rev(w, n):
    w = rev_step(w, n)
    for i in range(n + 1):
        print(f"=== {i} ===")
        print("\n".join(w))
        w = step(w)

test_rev(glider, 3)

# test_rev(parse(r"""
# ###.###
# #.#.#.#
# ###.###
# .......
# ###.###
# #.#.#.#
# ###.###
# """))


test_rev(parse(open(sys.argv[1]).read()), int(sys.argv[2]))
