#!/usr/bin/env bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name baseline_localism \
	--baseline \
	--n_dec_layers 8 \
	--n_enc_layers 8 \
	--n_heads 8 \
	--dropout 0.3 \
	--attention_dropout 0.1 \
	--optimizer "adam,lr=0.0001" \
	--emb_dim 512 \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_localism/fwd_9to21_train,data_localism/fwd_4to8_valid,data_localism/fwd_4to8_test" \
	--reload_size 10000000 \
	--epoch_size 125000 \
	--max_epoch 200  
echo done
