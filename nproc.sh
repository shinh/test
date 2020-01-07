#!/bin/bash

ncores=1

# `nproc` handles the case without a container including affinity.
nc=$(nproc 2> /dev/null)
if [[ $? == 0 ]]; then
    ncores="${nc}"
fi

# Take care of cgroup.
if [ -e /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    cpu_quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    if [[ "${cpu_quota}" != -1 ]]; then
        cpu_period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        ncores=$(("${cpu_quota}" / "${cpu_period}"))
    fi
fi

echo "${ncores}"
