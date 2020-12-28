ngc batch run \
    --name "ml.model-cnnf-clean-gn-block1" \
    --preempt RUNONCE \
    --ace nv-us-west-2 \
    --instance dgx1v.32g.8.norm \
    --image nvidia/pytorch:20.03-py3 \
    --result /results \
    --org nvidian --team nvr-aialgo \
    --datasetid 13254:/nvdatasets/imagenet \
    --datasetid 23145:/nvdatasets/imagenet_lmdb \
    --workspace UEaSlKO8SYWqAZK2nXR60Q:/cnnfresults:RW --workspace kpwhQIGrSTWmiRm1yg4mJg:/cnnfworkspace:RW \
    --port 1 --port 2 --port 1001 --port 1002 --port 2001 --port 2002 --port 3001 --port 3002 --port 4001 --port 4002 --port 5001 --port 5002 --port 6001 --port 6002 --port 7001 --port 7002 --port 8001 --port 8002 --port 9001 --port 9002 \
    --commandline "apt update; apt install screen -y; pip install advertorch; pip install tensorboardX; \
                bash -c 'sh /cnnfworkspace/Yujia/CNN-F/imagenet_apex/data_process/run.sh'; \
                bash -c 'cd /cnnfworkspace/Yujia/CNN-F/imagenet_apex/; python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
                train.py --max-cycles 1 \
                --layernorm 'gn' \
                --model 'block1' \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --lr 0.1 \
                --batch-size 32 \
                --epochs 90 \
                --scheduler-size 30 \
                --seed 0 \
                --grad-clip \
                --save-model 'imagenet_clean_gn_lr_0_1_res0_1_block1' \
                --model-dir '/cnnfresults/apex_imagenet/cleantraining' '"

