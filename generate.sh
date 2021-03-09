#!/bin/bash
python3 main.py --export_data true \
	--batch_size 2 \
	--cpu true \
	--exp_name fwd_data_gen \
	--num_workers 1 \
	--tasks prim_fwd \
	--env_base_seed -1 \
	--n_variables 1 \
	--n_coefficients 0 \
	--leaf_probs "0.75,0,0.25,0" \
	--max_ops 20 \
	--max_int 100 \
	--max_len 512 \
	--operators "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"

echo done
