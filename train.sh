#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm \
	--dropout 0.1 \
	--treelstm \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_precompute/fwd_train,data_precompute/fwd_valid,data_precompute/fwd_test" \
	--reload_size 100 \
	--epoch_size 100 \
	--max_epoch 1
echo done
