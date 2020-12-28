ngc batch run \
    --name "ml.model-cnnf-adv-gn-amp-resume" \
    --preempt RUNONCE \
    --ace nv-us-west-2 \
    --instance dgx1v.32g.8.norm \
    --image nvidia/pytorch:20.03-py3 \
    --result /results \
    --org nvidian --team nvr-aialgo \
    --datasetid 13254:/nvdatasets/imagenet \
    --datasetid 23145:/nvdatasets/imagenet_lmdb \
    --workspace UEaSlKO8SYWqAZK2nXR60Q:/cnnfresults:RW --workspace kpwhQIGrSTWmiRm1yg4mJg:/cnnfworkspace:RW \
    --total-runtime 120h --replicas 4 --port 8888 \
    --commandline "mpirun --allow-run-as-root -x NCCL_IB_HCA=mlx5_4,mlx5_6,mlx5_8,mlx5_10 -np \${NGC_ARRAY_SIZE} -npernode 1 \
                bash -c 'apt update; apt install screen -y; pip install advertorch; pip install tensorboardX; \
                sh /cnnfworkspace/Yujia/CNN-F/imagenet_apex/data_process/run.sh; \
                cd /cnnfworkspace/Yujia/CNN-F/imagenet_apex/; \
                python -m torch.distributed.launch --nproc_per_node=8 --nnodes=\${NGC_ARRAY_SIZE} \
                --node_rank=\${NGC_ARRAY_INDEX} --master_addr=\${NGC_MASTER_ADDR} \
                train.py --max-cycles 1 \
                --layernorm 'gn' \
                --model 'block1' \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --clean 'supclean' \
                --clean-parameter 0.05 \
                --lr 0.05 \
                --batch-size 32 \
                --eps 0.031 \
                --eps-iter 0.01 \
                --epochs 200 \
                --scheduler-size 50 \
                --seed 0 \
                --grad-clip \
                --adv-train \
                --use-amp \
                --resume '/cnnfresults/apex_imagenet/advtraining/imagenet_adv_amp_gn_block1_res_0_1_clean_0_05/checkpoint.pth.tar' \
                --save-model 'imagenet_adv_amp_gn_block1_res_0_1_clean_0_05' \
                --model-dir '/cnnfresults/apex_imagenet/advtraining' '"
