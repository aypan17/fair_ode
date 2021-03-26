#!/usr/bin/env bash
python3 main.py \
	--exp_name gcnn \
	--emb_dim 64 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--gcnn \
	--pad_tokens False \
	--optimizer "adam,lr=0.0001" \
	--batch_size 64 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,fwd_small/fwd_train,fwd_small/fwd_train,fwd_small/fwd_train" \
	--reload_size 500 \
	--epoch_size 20 \
	--max_epoch 1 \
        --validation_metrics valid_prim_fwd_acc 	
echo done
