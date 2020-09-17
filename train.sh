#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm_noparaminit \
	--emb_dim 8 \
	--cpu True \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--symmetric \
	--num_bit 10 \
	--optimizer "adam,lr=0.0001" \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_data "prim_fwd,prim_fwd.train,prim_fwd.valid,prim_fwd.test" \
	--reload_size 100 \
	--epoch_size 100 \
	--max_epoch 50 
echo done