#!/bin/bash

set -ex

mkdir -p $2
cp $1 $2
for i in $(ldd $1 | sed 's@[^/]*/@/@ ; s@\s.*@@'); do
    cp $i $2
done
