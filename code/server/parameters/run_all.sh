#!/usr/bin/env bash

sbatch parameters_DC.run
sbatch parameters_RR.run
bash run_Full.sh