#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm \
	--baseline \
	--n_dec_layers 6 \
	--n_heads 8 \
	--optimizer "adam,lr=0.0001" \
	--emb_dim 512 \
	--batch_size 20 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_precompute/fwd_train,data_precompute/fwd_valid,data_precompute/fwd_test" \
	--reload_size 100000 \
	--epoch_size 100000 \
	--max_epoch 50 
echo done
