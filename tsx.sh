#!/bin/sh
GLIBC=$HOME/src/glibc-tsx
GCC=/usr/local/stow/gcc-git/lib64
#export LD_DEBUG=all
LD_PRELOAD="$GLIBC/obj/libc.so $GLIBC/obj/nptl/libpthread.so $GCC/libgcc_s.so $GCC/libitm.so" LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu ~/src/glibc-tsx/obj/elf/ld-linux-x86-64.so.2 "$@"
