#!/usr/bin/env bash
ngc batch run \
    --name "baseline_localism" \
    --preempt RUNONCE \
    --ace nv-us-west-2 \
    --instance dgx1v.32g.8.norm \
	--image nvidia/pytorch:20.03-py3 \
	--datasetid 72879:/data \
	--commandline "export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
		--exp_name baseline_localism \
		--baseline \
		--n_dec_layers 8 \
		--n_enc_layers 8 \
		--n_heads 8 \
		--dump_path './cnnfresults/dumped/' \
		--dropout 0.3 \
		--optimizer 'adam,lr=0.0001' \
		--emb_dim 512 \
		--batch_size 32 \
		--tasks 'prim_fwd' \
		--reload_precomputed_data 'prim_fwd,data/fwd_9to21_train,data/fwd_4to8_valid,data/fwd_4to8_test' \
		--reload_size 10000000 \
		--epoch_size 125000 \
		--max_epoch 200  
	echo done"

