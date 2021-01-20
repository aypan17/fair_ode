#!/usr/bin/env bash
dim=(128 256)
enc_layers=(4 5 6 7 8)
dec_layers=(4 5 6 7 8)
drop=(0.0 0.1 0.2 0.3 0.4)
attn_drop=(0.0 0.1 0.2 0.3 0.4)
sym=(True False)
stack=('none' 'no-op' 'no-pop')
gate=(True False)
normalize=(True False)
size=(1 2 3 4 5)

export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name baseline \
	--baseline \
	--batch_size 32 \
	--tasks "prim_fwd" \
	--optimizer "adam,lr=0.0005" \
	--emb_dim ${dim[$(( $RANDOM % ${#dim[@]} ))]} \
	--dropout ${drop[$(( $RANDOM % ${#drop[@]} ))]} \
	--attention_dropout ${attn_drop[$(( $RANDOM % ${#attn_drop[@]} ))]} \
	--n_enc_layers ${enc_layers[$(( $RANDOM % ${#enc_layers[@]} ))]} \
	--n_dec_layers ${dec_layers[$(( $RANDOM % ${#dec_layers[@]} ))]} \
	--symmetric ${sym[$(( $RANDOM % ${#sym[@]} ))]} \
	--behavior ${stack[$(( $RANDOM % ${#stack[@]} ))]} \
	--gate_push_pop ${gate[$(( $RANDOM % ${#gate[@]} ))]} \
	--normalize_action ${normalize[$(( $RANDOM % ${#normalize[@]} ))]} \
	--stack_size ${size[$(( $RANDOM % ${#size[@]} ))]} \
	--reload_precomputed_data "prim_fwd,data_precompute/fwd_train,data_precompute/fwd_valid,data_precompute/fwd_test" \
	--reload_size 100000 \
	--epoch_size 12500 \
	--save_periodic 45 \
	--max_epoch 50
echo done
