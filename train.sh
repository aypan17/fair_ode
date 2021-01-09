#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm \
	--dropout 0.1 \
	--treelstm \
	--n_dec_layers 6 \
	--n_heads 8 \
	--emb_dim 8 \
	--batch_size 2 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_precompute/fwd_valid,data_precompute/fwd_valid,data_precompute/fwd_test" \
	--reload_size 10 \
	--epoch_size 10 \
	--max_epoch 1
echo done
