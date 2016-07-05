#!/bin/sh
#
# Usage:
#
# $ sh portable_crash.sh
# $ ARM_CC=arm-linux-gnueabihf-gcc-4.6 sh portable_crash.sh

# This code should crash if you remove the write call after #ifndef NDEBUG.
cat <<EOF > portable_crash.c
#include <stdio.h>
#include <unistd.h>

__attribute__((noinline))
void func(int x, int y, char* out) {
  char msg[] = "this write prevents the crash!\n";
#ifndef NDEBUG
  write(1, msg, sizeof(msg)-1);
#endif
  printf("die?%n\n");
  *out = x + y;
}

int main() {
  char buf[9999];
  func(1, 2, buf);
  return 0;
}
EOF

set -ex

run() {
    $@ -g portable_crash.c 2> /dev/null
    if ! ${RUNNER} ./a.out; then
        echo "shouldn't crash!"
        exit 1
    fi

    $@ -g -DNDEBUG portable_crash.c 2> /dev/null
    if ${RUNNER} ./a.out; then
        echo "should crash!"
        exit 1
    fi
}

os=$(uname)
for cc in gcc clang; do
    run ${cc}
    run ${cc} -O
    run ${cc} -m32
    run ${cc} -m32 -O
done

if [ x${ARM_CC} != x ]; then
    RUNNER="qemu-arm -L /usr/arm-linux-gnueabihf"
    run ${ARM_CC}
    run ${ARM_CC} -O
fi

echo OK!
