#!/usr/bin/env bash
python3 main.py \
	--exp_name data_gen \
	--emb_dim 8 \
	--cpu True \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--symmetric \
	--num_bit 10 \
	--optimizer "adam,lr=0.0001" \
	--batch_size 10 \
	--tasks "prim_fwd" \
	--export_data True \
	--reload_size 10 \
	--epoch_size 10 \
	--max_epoch 2 \
	--num_workers 1 
echo done
