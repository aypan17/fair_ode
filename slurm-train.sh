#!/bin/bash
#Submit this script with: sbatch <this-filename>

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres gpu:4 # number of GPUs per node
#SBATCH --qos normal #QOS to run in
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --ntasks-per-node=4 # tasks per node
#SBATCH -J lstm-transformer-noinit   # job name
#SBATCH --mail-user aypan@caltech.edu   # email address

#SBATCH -p tensortest # partition

# Notify on end and failure.
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

export NGPU=4; python3 -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
	--exp_name treelstm_noinit \
	--fp16 true --amp 2 \
	--master_port 8888 \
	--tasks "ode2" \
	--reload_data "ode2,ode2.train,ode2.valid,ode2.test" \
	--reload_size 100 \
	--emb_dim 256 \
	--n_dec_layers 6 \
	--n_heads 8 \
	--dropout 0.1 \
	--treelstm \
	--symmetric \
	--num_bit 10 \
	--optimizer "adam,lr=0.0001" \
	--batch_size 32 \
	--epoch_size 100 \
	--max_epoch 10 \
	--validation_metrics valid_ode2_acc
echo done
