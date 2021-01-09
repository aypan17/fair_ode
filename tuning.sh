#!/usr/bin/env bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name treernn \
	--treernn \
	--tune \
	--batch_size 128 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_precompute/fwd_train,data_precompute/fwd_valid,data_precompute/fwd_test" \
	--reload_size 2000000 \
	--epoch_size 2000000 \
	--save_periodic 25 \
	--max_epoch 30
echo done
