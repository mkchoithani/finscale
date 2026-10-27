#!/bin/bash
torchrun --standalone --nproc_per_node=1 train_gb10.py "$@"
