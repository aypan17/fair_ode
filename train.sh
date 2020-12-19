#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm \
	--emb_dim 128 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--gcnn \
	--pad_tokens False \
	--optimizer "adam,lr=0.0001" \
	--batch_size 64 \
	--tasks "prim_fwd" \
	--reload_data "prim_fwd,data.prefix.counts.train,data.prefix.counts.valid,data.prefix.counts.test" \
	--reload_size 400000 \
	--epoch_size 400000 \
	--max_epoch 50 \
        --validation_metrics valid_prim_fwd_acc 	
echo done
