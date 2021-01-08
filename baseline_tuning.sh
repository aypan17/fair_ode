#!/usr/bin/env bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name baseline \
	--baseline \
	--tune \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_data "prim_fwd,data/fwd_train.data,data/fwd_valid.data,data/fwd_test.data" \
	--reload_size 2000000 \
	--epoch_size 2000000 \
	--max_epoch 100
echo done