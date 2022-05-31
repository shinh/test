#!/usr/bin/env python3

import argparse


def parse_tsv(data):
    tbl = []
    for line in data.splitlines():
        tbl.append([c.strip() for c in line.split("\t")])
    return tbl


def show_md(tbl):
    def write_line(cols):
        print("| " + " | ".join(cols) + " |")

    for i, cols in enumerate(tbl):
        write_line(cols)
        if i == 0:
            write_line(["-"] * len(cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    args = parser.parse_args()

    with open(args.tsv) as f:
        tsv = f.read()

    tbl = parse_tsv(tsv)
    show_md(tbl)


if __name__ == "__main__":
    main()
