#!/usr/bin/env bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name baseline \
	--baseline \
	--tune \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_precompute/fwd_train,data_precompute/fwd_valid,data_precompute/fwd_test" \
	--reload_size 100000 \
	--epoch_size 100000 \
	--save_periodic 45 \
	--max_epoch 50
echo done
