#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm_noparaminit \
	--emb_dim 8 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--symmetric \
	--treernn \
	--character_rnn \
	--optimizer "adam,lr=0.0001" \
	--batch_size 256 \
	--tasks "prim_fwd" \
	--reload_data "prim_fwd,fwd_small/fwd_small.train,fwd_small/fwd_small.valid,fwd_small/fwd_small.test" \
	--reload_size 1000 \
	--epoch_size 1000 \
	--max_epoch 100 
echo done
