#!/usr/bin/env bash
export NGPU=2; python3 -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name treelstm_noparaminit \
	--emb_dim 256 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--treelstm \
	--optimizer "adam,lr=0.0001" \
	--batch_size 64 \
	--tasks "prim_fwd" \
	--reload_data "prim_fwd,data/fwd_train.data,data/fwd_valid.data,data/fwd_test.data" \
	--reload_size 100 \
	--epoch_size 100 \
	--max_epoch 5
echo done
