#!/usr/bin/env bash
python3 main.py \
	--exp_name treelstm_noparaminit \
	--emb_dim 256 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--treelstm \
	--symmetric \
	--num_bit 10 \
	--optimizer "adam,lr=0.0001" \
	--batch_size 32 \
	--tasks "ode2" \
	--reload_data "ode2,ode2.train,ode2.valid,ode2.test" \
	--reload_size 100000 \
	--epoch_size 100000 \
	--max_epoch 50 \
	--validation_metrics valid_ode2_acc  
echo done