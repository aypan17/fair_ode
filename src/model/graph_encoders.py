from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
"""
Implements models for the equation verification task.

Models are implemented using a custom batching forward pass. See `envs/char_sp.py` for graph format.
"""

import abc

import torch
import torch.nn as nn
import torch_scatter 

import time

import src.envs.char_sp as char_sp

from src.model.modules import MLP, UnaryLSTM, BinaryLSTM, BinaryLSTMSym, \
                    UnarySMU, BinarySMU, BinarySMUSym, UnaryStack, BinaryStack, BinaryStackSym

START_TOKEN = '<s>'
END_TOKEN = '</s>'


class TreeNN(torch.nn.Module):
    """
    A base class for unidirectional tree-structured models, which recursively compute
    the representation for each subtree from some function of the representation of the
    children of that subtree's root.

    This class makes the following assumptions about the model:
        - Leaves are always computed by an embedding lookup, and this may be done
          as the first step.
        - The root function is always the same.
        - Nodes pass fixed-size vectors along tree edges.
        - There is a fixed vocabulary for functions and tokens, known ahead of time.

    In exchange this class efficiently automates message passing for these models.

    Parameters
    ----------
    params
        The params used to initialize the model
    """
    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__()
        self.d_model = params.emb_dim
        self.id2word = id2word
        self.word2id = word2id
        self.una_ops = una_ops
        self.bin_ops = bin_ops
        self.pad_idx = params.pad_index
        self.leaf_emb = torch.nn.Embedding(
            num_embeddings=len(id2word),
            embedding_dim=params.emb_dim,
            padding_idx=self.pad_idx,
            max_norm=1.0, # is this needed?
        )
        self.num_enc = torch.nn.Sequential(
            nn.Linear(1, params.emb_dim),
            nn.Sigmoid(),
            nn.Linear(params.emb_dim, params.emb_dim),
            nn.Sigmoid()
        )

    @abc.abstractmethod
    def _apply_function(self, function_name: str, input_cell: torch.Tensor, input_hidden: torch.Tensor, train) -> torch.Tensor:
        """
        Compute the activation for an internal node of the tree.

        Parameters
        ----------
        function_name
            Name of the function to apply.
        inputs
            Inputs to each module in the current step.
            Has shape [batch_size, function_arity, self.hparams.d_model],
            where `batch_size` is the number of nodes being computed at this step.

        Returns
        -------
        torch.Tensor
            The vector to pass up the tree.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_output(self, inputs, lens):
        """
        Compute the outputs for this model.

        Parameters
        ----------
        inputs
            Inputs to each root module in the batch.
            Has shape [batch_size, root_function_arity, self.hparams.d_model].
        lens
            Number of embeddings/nodes of each graph in batch.
            Has shape [batch_size].
        pad
            Whether to pad the output with start and end tokens.

        Returns
        -------
        torch.Tensor
            Whatever output this model produces.
        """
        batch = torch.split(inputs, lens.tolist(), dim=0)
        return pad_sequence(batch, padding_value=0.0, batch_first=True).squeeze(2)
        '''
    def forward(self, x, lengths, train=False):
        """
        Given a (possibly batched) graph and the number of nodes per tree, 
        compute the outputs for each tree.
        """
        s = time.time()
        operations, tokens, left_idx, right_idx, depths, operation_order = x
        num_steps = len(operation_order)#operation_order.numel()
        num_nodes = operations.numel()

        # A buffer where the i-th row is the activations output from the i-th node
        # The buffer is repeatedly summed into to allow dense gradient computation;
        # this is valid because each position is summed to exactly once.
        activations = torch.zeros(
            (num_nodes, self.d_model), device=tokens.device
        )

        # A buffer where the i-th row is the memory output from the i-th node.
        memory = torch.zeros(
            (num_nodes, self.memory_size, self.d_model), device=tokens.device
        )

        for depth in range(num_steps):  
            step_mask = depths == depth  # Indices to compute at this step
            op = operation_order[depth]#.item()

            if op == -1: # Embedding lookup or number encoding
                leaf_tokens = tokens.masked_select(step_mask)
                step_activations = self.leaf_emb(leaf_tokens)
                activations = activations.masked_scatter(
                    step_mask.unsqueeze(1), step_activations
                )
            else:
                op_name = self.id2word[op]
                step_ids = torch.nonzero(step_mask, as_tuple=True)[0]
                left = left_idx.masked_select(step_mask)
                right = right_idx.masked_select(step_mask) # if we have a unary function this is meaningless
                predecessors = torch.stack((left, right), dim=1)
                input_cell = activations[predecessors]
                input_hidden = memory[predecessors]

                step_activations, step_memory = self._apply_function(
                    op_name, input_cell, input_hidden, train
                )

                activation_scatter = torch_scatter.scatter(
                    src=step_activations, index=step_ids, dim=0, dim_size=num_nodes
                )
                memory_scatter = torch_scatter.scatter(
                    src=step_memory, index=step_ids, dim=0, dim_size=num_nodes
                )
                activations = activations + activation_scatter
                memory = memory + memory_scatter
        e = time.time()
        print("fwd", e-s)
        # Reverse activations because nodes are listed in reverse pre-order.
        return self._compute_output(activations, lengths)
       '''
    
    def forward(self, x, lengths, train=False):
        """
        Given a (possibly batched) graph and the number of nodes per tree, 
        compute the outputs for each tree.
        """
        s = time.time()
        operations, tokens, left_idx, right_idx, depths, operation_order = x
        num_steps = len(operation_order)
        num_nodes = operations.numel()

        # A buffer where the i-th row is the activations output from the i-th node
        # The buffer is repeatedly summed into to allow dense gradient computation;
        # this is valid because each position is summed to exactly once.
        activations = torch.zeros(
            (num_nodes, self.d_model), device=tokens.device
        )

        # A buffer where the i-th row is the memory output from the i-th node.
        memory = torch.zeros(
            (num_nodes, self.memory_size, self.d_model), device=tokens.device
        )
        fwd = 0
        idx = torch.stack((left_idx, right_idx), dim=1)
        for depth in range(num_steps):  
            step_mask = (depths == depth).unsqueeze(1)  # Indices to compute at this step
            op = operation_order[depth]

            if op == -1: # Embedding lookup or number encoding
                activations += self.leaf_emb(tokens) * step_mask

            else:
                op_name = self.id2word[op]                
                inp = activations[idx]
                mem = memory[idx]
                s1 = time.time()
                step_activations, step_memory = self._apply_function(
                    op_name, inp, mem, train
                )
                e1 = time.time()
                fwd += e1-s1
                activations = activations + step_activations * step_mask
                memory = memory + step_memory * step_mask.unsqueeze(1)
        #print("fwd", fwd)
        e = time.time()
        #print(e-s)
        # Reverse activations because nodes are listed in reverse pre-order.
        return self._compute_output(activations, lengths)
    
    
class TreeRNN(TreeNN):
    """
    A TreeRNN model.

    For full parameters, see the docstring for `TreeNN`.

    Parameters
    ----------
    num_module_layers
        How many layers to use for each internal module.
    """

    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__(params, id2word, word2id, una_ops, bin_ops)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: MLP(1, params.emb_dim, params.num_module_layers, params.tree_activation) for f in una_ops}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: MLP(2, params.emb_dim, params.num_module_layers, params.tree_activation) for f in bin_ops}
        )
        if params.symmetric:   
            self.binary_function_modules['add'] = \
                MLP(1, params.emb_dim, params.num_module_layers, params.tree_activation)
            self.binary_function_modules['mul'] = \
                MLP(1, params.emb_dim, params.num_module_layers, params.tree_activation)
        self.symmetric = params.symmetric
        self.memory_size = 1

    def _apply_function(self, function_name, inputs, memory, train):
        if function_name in self.una_ops:
            module = self.unary_function_modules[function_name]
            inputs = inputs[:, 0, :] 
            return module(inputs, train), memory[:, 0, :]

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            if self.symmetric and function_name in ['add', 'mul']:
                inputs_together = inputs[:, 0, :] + inputs[:, 1, :]
            else:
                inputs_together = inputs.view(inputs.size(0), -1)

            return module(inputs_together, train), memory[:, 0, :]

        assert False

class TreeLSTM(TreeNN):
    """
    A TreeLSTM model.

    For full parameters, see the docstring for `TreeNN`.
    """

    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__(params, id2word, word2id, una_ops, bin_ops)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: UnaryLSTM(self.d_model, params.dropout) for f in una_ops}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: BinaryLSTM(self.d_model, params.dropout) for f in bin_ops}
        )
        if params.symmetric:   
            self.binary_function_modules['add'] = BinaryLSTMSym(self.d_model, params.dropout)
            self.binary_function_modules['mul'] = BinaryLSTMSym(self.d_model, params.dropout)
        self.memory_size = 1

    def _apply_function(self, function_name, inputs, memory, train):
        if function_name in self.una_ops:
            module = self.unary_function_modules[function_name]
            inputs = inputs[:, 0, :]
            memory = memory[:, 0, :]
            return module(inputs, memory, train)

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            return module(l_inputs, r_inputs, l_memory, r_memory, train)

        assert False


class TreeSMU(TreeNN):
    """
    A TreeSMU model.

    For full parameters, see the docstring for `TreeNN`.
    """

    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__(params, id2word, word2id, una_ops, bin_ops)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: UnarySMU(params) for f in una_ops}
        )
        self.unary_stack_modules = torch.nn.ModuleDict(
            {f: UnaryStack(params) for f in una_ops}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: BinarySMU(params) for f in bin_ops}
        )
        self.binary_stack_modules = torch.nn.ModuleDict(
            {f: BinaryStack(params) for f in bin_ops}
        )
        if params.symmetric:
            self.binary_function_modules['add'] = BinarySMUSym(params)
            self.binary_function_modules['mul'] = BinarySMUSym(params)
            self.binary_stack_modules['add'] = BinaryStackSym(params)
            self.binary_stack_modules['mul'] = BinaryStackSym(params)
        self.memory_size = params.stack_size

    def _apply_function(self, function_name, inputs, memory, train):
        if function_name in self.una_ops:
            module = self.unary_function_modules[function_name]
            stack = self.unary_stack_modules[function_name]
            inputs = inputs[:, 0, :]
            memory = memory[:, 0, :]
            step_memory = stack(inputs, memory, train)
            return module(inputs, step_memory).squeeze(1), step_memory

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            stack = self.binary_stack_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            step_memory = stack(l_inputs, r_inputs, l_memory, r_memory, train)
            return module(l_inputs, r_inputs, step_memory).squeeze(1), step_memory

        assert False