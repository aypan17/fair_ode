#!/bin/bash
#Submit this script with: sbatch <this-filename>

#SBATCH --time=45:00:00   # walltime
#SBATCH -n 28   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --qos normal #QOS to run in
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J fwd_data_gen   # job name
#SBATCH --mail-user aypan@caltech.edu   # email address

#SBATCH -p tensortest # partition

# Notify on end and failure.
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

python3 main.py --export_data true \
	--batch_size 32 \
	--cpu true \
	--exp_name fwd_data_gen \
	--num_workers 28 \
	--tasks prim_fwd \
	--env_base_seed -1 \
	--n_variables 1 \
	--n_coefficients 0 \
	--leaf_probs "0.75,0,0.25,0" \
	--max_ops 20 \
	--max_int 5 \
	--max_len 512 \
	--master_port 12001 \
	--operators "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"

echo done
