#!/bin/sh

set -ex

#gcc -m32 -nostartfiles loader.c -static -o loader
gcc -m32 loader.c -static -o loader
#gcc -m32 -Wl,-Ttext=0x100000 -Wl,-Tdata=0x200000 loader.c -static -o loader
gcc -m32 -Wl,--dynamic-linker=$(pwd)/loader hello.c -o hello-loader
./hello-loader
