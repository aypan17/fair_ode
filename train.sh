#!/usr/bin/env bash
export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name treelstm \
	--treelstm \
	--n_dec_layers 6 \
	--n_heads 8 \
	--optimizer "adam,lr=0.0005" \
	--emb_dim 8 \
	--batch_size 1 \
	--tasks "prim_fwd" \
	--reload_precomputed_data 'prim_fwd,data_precompute/fwd_train,data_precompute/fwd_valid,data_precompute/fwd_test' \
	--reload_size 10 \
	--epoch_size 2 \
	--max_epoch 1  
echo done
