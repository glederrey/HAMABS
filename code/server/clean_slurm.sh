#!/usr/bin/env bash

for i in $(ls -d */);
do
    cd $i
    rm slurm*
    cd ..
done