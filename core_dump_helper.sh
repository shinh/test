#!/bin/sh
#
# A core dump helper
#
# sudo sh -c 'echo "| /home/hamaji/test/core_dump_helper.sh /tmp/core.%p" > /proc/sys/kernel/core_pattern'
#

cat > $1
chmod 644 $1

