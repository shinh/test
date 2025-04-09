#!/usr/bin/python3

import argparse
import collections
import glob
import logging
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=str, nargs="+")
    args = parser.parse_args()

    size_by_basename = collections.defaultdict(int)
    size_by_dirname = collections.defaultdict(int)
    size_by_subdirname = collections.defaultdict(int)
    total = 0

    logging.basicConfig(
        format="%(levelname)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.getLevelName("INFO"),
    )

    all_filenames = []
    for target_dir in args.dirs:
        logging.info(f"Glob for {target_dir}")
        filenames = list(glob.glob(os.path.join(target_dir, "**/*"), recursive=True))
        logging.info(f"{target_dir} has {len(filenames)} files")
        for filename in filenames:
            all_filenames.append((target_dir, filename))

    for base_dir, filename in all_filenames:
        size = os.path.getsize(filename)
        total += size
        size_by_basename[os.path.basename(filename)] += size
        dirname = os.path.dirname(os.path.relpath(filename, base_dir))
        size_by_subdirname[os.path.basename(dirname)] += size
        while dirname:
            size_by_dirname[dirname] += size
            dirname = os.path.dirname(dirname)

    def show_stats(stats):
        for name, size in sorted(
                stats.items(),
                key=lambda s: s[1],
                reverse=True):
            percent = size / total * 100
            print(f"  {name}: {size / 1024 / 1024:.1f} MB {percent:.1f}%")

    print("File sizes by basename:")
    show_stats(size_by_basename)
    print()

    print("File sizes by subdirname:")
    show_stats(size_by_subdirname)
    print()

    print("File sizes by dirname:")
    show_stats(size_by_dirname)
    print()


if __name__ == "__main__":
    main()
