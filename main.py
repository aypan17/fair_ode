# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random
import argparse
import numpy as np
import torch

import src
from src.slurm import init_signal_handler, init_distributed_mode, init_ngc_job
from src.utils import bool_flag, initialize_exp
from src.model import check_model_params, build_modules
from src.envs import ENVS, build_env
from src.trainer import Trainer
from src.evaluator import Evaluator


np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=256,
                        help="Embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    
    # encoder parameters
    parser.add_argument('--symmetric', action='store_true',
                        help='run the version with enforced symmetry on add and mul')
    parser.add_argument('--character_rnn', action='store_true',
                        help='use a character RNN to encode numbers')
    parser.add_argument('--pad_tokens', type=bool, default=False,
                        help='pad the output of the encoder with start and end token embeddings')
    parser.add_argument('--treernn', action='store_true',
                        help='use a TreeRNN encoder for the model')
    parser.add_argument('--treelstm', action='store_true',
                        help='use a TreeLSTM encoder for the model')
    parser.add_argument('--gcnn', action='store_true',
                        help='use a Graph CNN encoder for the model')
    parser.add_argument('--baseline', action='store_true',
                        help='User a Transformer encoder for the model')
    parser.add_argument('--num_module_layers', type=int, default=2,
                        help='number of layers per module in the RNN/GCNN model')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of message passing steps in the GCNN model')


    # smu parameters
    parser.add_argument('--treesmu', action='store_true',
                        help='use a TreeSMU encoder for the model')
    parser.add_argument('--stack_size', type=int, default=5,
                        help='max size of the stack/queue')
    parser.add_argument('--tree_activation', type=str, default='tanh',
                        help='tree node activation')
    parser.add_argument('--stack_activation', type=str, default='tanh',
                        help='stack node activation')
    parser.add_argument('--no_op', default=False, action='store_true',
                        help='add no-op as an additional stack op')
    parser.add_argument('--no_pop', default=False, action='store_true',
                        help='add only push and no-op')
    parser.add_argument('--like_LSTM', default=False, action='store_true',
                        help='make the mem2out stack tree behave as an LSTM+stack')
    parser.add_argument('--gate_push_pop', default=False, action='store_true',
                        help='make the push pop action a gate rather than a number')
    parser.add_argument('--normalize_action', default=False, action='store_true',
                        help='normalize push-pop magnitude before push and pop')
    parser.add_argument('--gate_top_k', default=False, action='store_true',
                        help='gate the top-k instead of weighted average')
    parser.add_argument('--top_k', type=int, default=1,
                        help='use the top-k stack elements to compute the output.')


    # training parameters
    parser.add_argument("--tune", action='store_true',
                        help="Run hyperparameter tuning")
    parser.add_argument("--env_base_seed", type=int, default=0,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum sequences length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=300000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")
    parser.add_argument("--same_nb_ops_per_batch", type=bool_flag, default=False,
                        help="Generate sequences with the same number of operators in batches.")

    # export data / reload it
    parser.add_argument("--export_data", type=bool_flag, default=False,
                        help="Export data and disable training.")
    parser.add_argument("--reload_data", type=str, default="",
                        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)")
    parser.add_argument("--reload_precomputed_data", type=str, default="",
                        help="Load precomputed tensors from the disk (task1,train_path_prefix,valid_path_prefix,test_path_prefix")
    parser.add_argument("--reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--precompute_tensors", type=str, default="",
                        help="Precompute the given dataset for faster tree-equation training")
    parser.add_argument("--compute_augs", type=bool_flag, default=False,
                        help="Generate augmentations of the precomputed tensors. Only works for precompute_tensors")


    # environment parameters
    parser.add_argument("--env_name", type=str, default="char_sp",
                        help="Environment name")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="",
                        help="Tasks")
    parser.add_argument("--order", type=int, default=1,
                        help="Tasks")
    parser.add_argument("--vars", type=int, default=1,
                        help="Tasks")


    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=False,
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--ngc", action="store_true",
                        help="Initiate job with ngc.py file; otherwise use default slurm.py")

    # ------------------ Apex Related ------------------
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    #parser.add_argument('--local_rank', default=-1, type=int, help='process rank for distributed training')
    parser.add_argument('--node-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument("--apex_syncbn", action="store_true",
                        help="learning with apex synchronized batch norm")

    return parser

def tune(params):
    # Discrete hparam range
    dim = [128, 256, 512]
    dec_layers = [4, 5, 6, 7, 8]
    heads = [4, 8, 16]
    sin_emb = [True, False]
    params.emb_dim = np.random.choice(dim)
    params.num_dec_layers = np.random.choice(dec_layers)
    params.n_heads = np.random.choice(heads)
    params.sinusoidal_embeddings = np.random.choice(sin_emb)

    # Continuous hparam range
    lr = [0.00003, 0.005]
    drop = [0, 0.4]
    attn_drop = [0, 0.4]
    params.optimizer = "adam,lr="+str(np.random.uniform(lr[0], lr[1]))
    params.dropout = np.random.uniform(drop[0], drop[1])
    params.attention_dropout = np.random.uniform(attn_drop[0], attn_drop[1])

    print(f"Tuning dim from: {dim}")
    print(f"Tuning num_dec_layers from: {dec_layers}")
    print(f"Tuning num_heads from: {heads}")
    print(f"Tuning sin_emb from: {sin_emb}")
    print("============================================")
    print(f"Tuning learning rate from the interval: {lr}")
    print(f"Tuning dropout from the interval: {drop}")
    print(f"Tuning attention dropout from the interval: {attn_drop}")

    if params.tree_enc:
        #pad_tokens = [True, False]
        #params.pad_tokens = np.random.choice(pad_tokens)
        #print(f"Tuning pad_tokens from: {pad_tokens}")
        params.symmetric = np.random.choice([True, False])

        if params.treesmu:
            stack_behavior = np.random.choice(['none', 'no-op', 'no-pop'])
            if stack_behavior == 'no-op':
                params.no_op = True
            if stack_behavior == 'no-pop':
                params.no_pop = True
            params.gate_push_pop = np.random.choice([True, False])
            params.normalize_action = np.random.choice([True, False])
            params.top_k = np.random.choice([1, 2, 3, 4, 5])
            params.gate_top_k = np.random.choice([True, False]) if params.top_k > 1 else False
            params.stack_size = np.random.choice(range(params.top_k, 6))
            print(f"Tuning stack_size from: {params.top_k} to 5")

    elif params.baseline:
        enc_layers = [4, 5, 6, 7, 8]
        params.num_enc_layers = np.random.choice(enc_layers)
        print(f"Tuning num_enc_layers from: {enc_layers}")
    else:
        assert False


def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    if params.ngc:
        init_ngc_job(params)
    else:
        init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()

    # CPU / CUDA
    if params.cpu:
        assert not params.multi_gpu
    else:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # write word2id dictionaries if precomputing
    if params.precompute_tensors:
        assert len(params.tasks) == 1
        trainer.file_handler_data.write(json.dumps(env.word2id))
        trainer.file_handler_data.write("\n")
        trainer.file_handler_data.write(json.dumps(env.una_ops))
        trainer.file_handler_data.write("\n")
        trainer.file_handler_data.write(json.dumps(env.bin_ops))
        trainer.file_handler_data.write("\n")
        trainer.file_handler_data.flush()
        trainer.precompute_tensors()
        exit()

    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0

        while trainer.n_equations < trainer.epoch_size:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    trainer.enc_dec_step(task)
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        torch.cuda.empty_cache()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    assert sum([params.treernn, params.treelstm, params.treesmu, params.baseline]) == 1
    params.tree_enc = not params.baseline

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True

    if params.tune:
        tune(params)

    # check parameters
    check_model_params(params)
    # run experiment
    main(params)
