#!/usr/bin/env bash
python3 main.py \
	--exp_name precompute \
	--tasks "prim_fwd" \
	--precompute_tensors "data/fwd_test.data"
echo done
