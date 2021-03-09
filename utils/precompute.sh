#!/usr/bin/env bash
python3 main.py \
	--exp_name precompute \
	--tasks "prim_fwd" \
	--precompute_tensors "data_sorted/split_ai"
echo done
