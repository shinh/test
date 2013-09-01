#!/bin/sh
GLIBC=$HOME/src/glibc-tsx
GCC=/usr/local/stow/gcc-git/lib32
#export LD_DEBUG=all
LD_PRELOAD="$GCC/libgcc_s.so $GCC/libitm.so" "$@"
