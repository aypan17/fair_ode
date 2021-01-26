#!/usr/bin/env bash
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name lstm_localism \
	--treelstm \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.3 \
	--attention_dropout 0.2 \
	--optimizer "adam,lr=0.0001" \
	--emb_dim 256 \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--reload_precomputed_data "prim_fwd,data_localism/fwd_9to21_train,data_localism/fwd_4to8_valid,data_localism/fwd_4to8_test" \
	--reload_size 10000000 \
	--epoch_size 125000 \
	--max_epoch 200  
echo done