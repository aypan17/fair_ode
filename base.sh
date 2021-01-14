#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm \
	--emb_dim 256 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--treelstm \
	--optimizer "adam,lr=0.0001" \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_data "prim_fwd,data/fwd_valid.data,data/fwd_valid.data,data/fwd_test.data" \
	--reload_size 100000 \
	--epoch_size 2000 \
	--max_epoch 1
echo done
