from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
"""
Implements models for the equation verification task.

Models are implemented using DGL. See `envs/char_sp.py` for graph format.
"""

import abc

import dgl
import dgl.nn
import torch
import torch_scatter

from .modules import FunctionModule, UnaryLSTM, BinaryLSTM, \
                    UnarySMU, BinarySMU, UnaryStack, BinaryStack
import src.envs.char_sp as char_sp

START_TOKEN = '<s>'
END_TOKEN = '</s>'

def pad_tokens(model, out):
	"""
	Pads the front and back of the string with embeddings for <s> and </s>
	from the model

	Parameters
	----------
	model
		The encoder model calling this function.
	out
		The unpadded output of the encoder model.

	Returns
	-------
		Output from the model padded with embeddings for <s> and </s>.
	"""
	batch_size = len(out)
	# (batch_size, 1, d_model)
	s = torch.tensor([model.token_vocab[START_TOKEN]], device=out[0].device)
	e = torch.tensor([model.token_vocab[END_TOKEN]], device=out[0].device)
	start = model.embedding(s)
	end = model.embedding(e)
	# This is slow! Looping is slow but not sure how to concatenated to variable length tensors.
	return [torch.cat((start, o, end), dim=0) for o in out]


class GraphClassifier(torch.nn.Module, abc.ABC):
    """
    A superclass for models which will be used for binary classification on graphs.

    Parameters
    ----------
    bin_ops
        List of all binary functions in vocabulary.
    una_ops
        List of all binary functions in vocabulary.
    function_vocab
        Vocabulary mapping function names to indices.
    token_vocab
        Vocabulary mapping token names to indices.
    """

    def __init__(self, bin_ops, una_ops, function_vocab, token_vocab):
        super().__init__()
        self.bin_ops = bin_ops
        self.una_ops = una_ops
        self.function_vocab = function_vocab
        self.function_vocab_inverse = {v: k for k, v in function_vocab.items()}
        self.token_vocab = token_vocab
        self.token_vocab_inverse = {v: k for k, v in token_vocab.items()}

    @abc.abstractmethod
    def forward(self, graph, lengths):
        """
        Given a DGLGraph (possibly batched) and the number of nodes, produce a encoded list of embeddings
        """
        raise NotImplementedError


class RecursiveNN(GraphClassifier):
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
    d_model
        The dimension of the vectors passed along edges of the tree.
    """

    def __init__(self, d_model, bin_ops, una_ops, function_vocab, token_vocab):
        super().__init__(bin_ops, una_ops, function_vocab, token_vocab)

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(token_vocab),
            embedding_dim=d_model,
            padding_idx=token_vocab[char_sp.NONLEAF_TOKEN],
        )

        self.d_model = d_model

    @abc.abstractmethod
    def _compute_output(self, inputs, lens, pad):
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
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_function(self, function_name, input_cell, input_hidden):
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

    @staticmethod
    def _compute_predecessors(node_group, predecessor_index):
        """
        Given a batch of nodes indices for the same function and an index
        mapping nodes to their predecessors (with placeholders), return just
        the predecessors (without placeholders).

        Parameters
        ----------
        node_group
            A batch of nodes [shape: num_nodes], all of which must have the same arity.
        predecessor_index
            An index (shape: [num_all_nodes, max_in_degree]) mapping nodes to their
            predecessors, which may have placeholders.
        """
        preds_with_placeholders = predecessor_index[node_group]

        # Clip off the placeholders, assuming every row has the same number of them
        arity = (preds_with_placeholders[0] != char_sp.INDEX_PLACEHOLDER).sum().item()
        predecessors = preds_with_placeholders.narrow(dim=1, start=0, length=arity)

        return predecessors

    def forward(self, graph, lengths, pad=False):
        """
        Given a (possibly batched) graph and the number of nodes per tree, 
        compute the outputs for each tree.
        """
        num_nodes = graph.num_nodes()
        leaf_mask = (
            graph.ndata[char_sp.FUNCTION_FIELD]
            == self.function_vocab[char_sp.LEAF_FUNCTION]
        )
        # Our graphs do not currently have equality function.
        '''
        root_mask = (
            graph.ndata[char_sp.FUNCTION_FIELD]
            == self.function_vocab[char_sp.EQUALITY_FUNCTION]
        )
        '''
        internal_mask = ~leaf_mask
        internal_order = char_sp.typed_topological_nodes_generator(
            graph, node_mask=internal_mask
        )
        predecessor_index = char_sp.tensorize_predecessors(graph)

        # A buffer where the i-th row is the activations output from the i-th node
        # The buffer is repeatedly summed into to allow dense gradient computation;
        # this is valid because each position is summed to exactly once.
        activations = torch.zeros(
            (num_nodes, self.d_model), device=graph.device
        )

        # A buffer where the i-th row is the memory output from the i-th node.
        memory = torch.zeros(
            (num_nodes, self.memory_size, self.d_model), device=graph.device
        )

        # Precompute all leaf nodes at once
        tokens = graph.ndata[char_sp.TOKEN_FIELD][leaf_mask]
        token_activations = self.embedding(tokens)
        activations = activations.masked_scatter(
            leaf_mask.unsqueeze(1), token_activations
        )

        # Compute all internal nodes under the topological order
        for node_group in internal_order:
            # Look up the type of the first node, since they should all be the same
            function_idx = graph.ndata[char_sp.FUNCTION_FIELD][node_group[0]].item()
            function = self.function_vocab_inverse[function_idx]

            # Gather inputs, call the module polymorphically, and scatter to buffer
            predecessors = self._compute_predecessors(node_group, predecessor_index)
            input_cell = activations[predecessors]
            input_hidden = memory[predecessors]
            step_activations, step_memory = self._apply_function(
                function, input_cell, input_hidden
            )
            activation_scatter = torch_scatter.scatter(
                src=step_activations, index=node_group, dim=0, dim_size=num_nodes
            )
            memory_scatter = torch_scatter.scatter(
                src=step_memory, index=node_group, dim=0, dim_size=num_nodes
            )
            activations = activations + activation_scatter
            memory = memory + memory_scatter

        # Reverse activations because nodes are listed in reverse pre-order.
        return self._compute_output(torch.flip(activations, dims=[0]), lengths, pad)
        

class TreeRNN(RecursiveNN):
    """
    A TreeRNN model.

    For full parameters, see the docstring for `RecursiveNN`.

    Parameters
    ----------
    num_module_layers
        How many layers to use for each internal module.
    """

    def __init__(self, d_model, num_module_layers, una_ops, bin_ops, function_vocab, token_vocab):
        super().__init__(d_model, bin_ops, una_ops, function_vocab, token_vocab)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: FunctionModule(1, d_model, num_module_layers) for f in una_ops + ['INT+', 'INT-']}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: FunctionModule(2, d_model, num_module_layers) for f in bin_ops}
        )
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.memory_size = 1

    def _compute_output(self, inputs, lens, pad):
        unpadded_batch = torch.split(inputs, lens.tolist()[::-1], dim=0)[::-1]
        batch = pad_tokens(self, unpadded_batch) if pad else unpadded_batch
        return pad_sequence(batch, padding_value=0.0, batch_first=True).squeeze(2)

    def _apply_function(self, function_name, inputs, memory):
        if function_name in (self.una_ops + ['INT+', 'INT-']):
            module = self.unary_function_modules[function_name]
            # Child-sum
            inputs = inputs[:, 0, :] if function_name in self.una_ops else torch.sum(inputs, dim=1)
            return module(inputs), memory[:, 0, :]

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            inputs_together = inputs.view(inputs.size(0), -1)
            return module(inputs_together), memory[:, 0, :]

        assert False


class TreeLSTM(RecursiveNN):
    """
    A TreeLSTM model.

    For full parameters, see the docstring for `RecursiveNN`.
    """

    def __init__(self, d_model, una_ops, bin_ops, function_vocab, token_vocab):
        super().__init__(d_model, bin_ops, una_ops, function_vocab, token_vocab)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: UnaryLSTM(d_model) for f in una_ops + ['INT+', 'INT-']}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: BinaryLSTM(d_model) for f in bin_ops}
        )
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.memory_size = 1

    def _compute_output(self, inputs, lens, pad):
        unpadded_batch = torch.split(inputs, lens.tolist()[::-1], dim=0)[::-1]
        batch = pad_tokens(self, unpadded_batch) if pad else unpadded_batch
        return pad_sequence(batch, padding_value=0.0, batch_first=True).squeeze(2)

    def _apply_function(self, function_name, inputs, memory):
        if function_name in (self.una_ops + ['INT+', 'INT-']):
            module = self.unary_function_modules[function_name]
            # Child-sum
            inputs = inputs[:, 0, :] if function_name in self.una_ops else torch.sum(inputs, dim=1)
            memory = memory[:, 0, :]
            return module(inputs, memory)

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            return module(l_inputs, r_inputs, l_memory, r_memory)

        assert False

class TreeSMU(RecursiveNN):
    """
    A TreeSMU model.

    For full parameters, see the docstring for `RecursiveNN`.
    # TODO: implement
    """

    def __init__(self, params, una_ops, bin_ops, function_vocab, token_vocab):
        super().__init__(params.emb_dim, bin_ops, una_ops, function_vocab, token_vocab)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: UnarySMU(params) for f in una_ops + ['INT+', 'INT-']}
        )
        self.unary_stack_modules = torch.nn.ModuleDict(
            {f: UnaryStack(params) for f in una_ops + ['INT+', 'INT-']}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: BinarySMU(params) for f in bin_ops}
        )
        self.binary_stack_modules = torch.nn.ModuleDict(
            {f: BinaryStack(params) for f in bin_ops}
        )
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.memory_size = params.stack_size

    def _compute_output(self, inputs, lens, pad):
        unpadded_batch = torch.split(inputs, lens.tolist()[::-1], dim=0)[::-1]
        batch = pad_tokens(self, unpadded_batch) if pad else unpadded_batch
        return pad_sequence(batch, padding_value=0.0, batch_first=True).squeeze(2)

    def _apply_function(self, function_name, inputs, memory):
        if function_name in (self.una_ops + ['INT+', 'INT-']):
            module = self.unary_function_modules[function_name]
            stack = self.unary_stack_modules[function_name]
            # Child-sum
            inputs = inputs[:, 0, :] if function_name in self.una_ops else torch.sum(inputs, dim=1)
            memory = memory[:, 0, :]
            step_memory = stack(inputs, memory)
            return module(inputs, step_memory).squeeze(1), step_memory

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            stack = self.binary_stack_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            step_memory = stack(l_inputs, r_inputs, l_memory, r_memory)
            return module(l_inputs, r_inputs, step_memory).squeeze(1), step_memory

        assert False


class GraphCNN(GraphClassifier):
    """
    A graph convolutional model for equation verification.

    First, every node gets a feature from an embedding lookup on its function or token.
    Then, for each layer, nodes update their features by applying a module
    (shared across all nodes) to four inputs concatenated in order:
        - The left child's features
        - The right child's features
        - The parent's features
        - The node's own features
    If any of these are not present, zeroes are used instead.

    Each expression (left or right of an Equality) is processed separately.
    Then, the expression is represented by its mean feature vector, and the logit
    of the equation is the dot product of the left and right mean vectors.


    Parameters
    ----------
    d_model
        Dimension of the embedding vectors and features.
    num_layers
        Number of times to perform message-passing between nodes.
    num_module_layers
        Number of layers to use in each module for message aggregation.

    See GraphClassifier for other parameter descriptions.
    """

    def __init__(self, d_model, num_layers, num_module_layers, una_ops, bin_ops, function_vocab, token_vocab):
        super().__init__(bin_ops, una_ops, function_vocab, token_vocab)

        # A single vocab for both functions and tokens
        self.d_model = d_model
        self.min_function_idx = len(token_vocab)
        self.combined_vocab = {
            **token_vocab,
            **{
                name: idx + self.min_function_idx
                for name, idx in function_vocab.items()
            },
        }
        self.combined_vocab_inverse = {
            v: k for k, v in self.combined_vocab.items()
        }

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.combined_vocab),
            embedding_dim=d_model,
            padding_idx=self.min_function_idx
            + self.function_vocab[char_sp.PAD_FUNCTION],
        )

        self.layers = torch.nn.ModuleList(
            [FunctionModule(4, d_model, num_module_layers) for _ in range(num_layers)]
        )

        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(1))

    @staticmethod
    def _reduce_layer(layer):
        """
        Make a DGL user-defined reduction function from a module.

        The function expects a node batch with features of shape
        [n_nodes, n_arguments, dim], and applies a module which is expected to accept
        tensors of shape n_arguments * dim to produce new features of shape dim.

        Parameters
        ----------
        layer
            The module to apply during reduction. Must be polymorphic in its first
            tensor dimension (i.e. batch-size agnostic).

        Returns
        -------
        NodeBatch -> Dict[str, Tensor]
            A user-defined reduction function which applies `layer` to the flattened
            input messages.
        """

        def layer_udf(nodes):
            batch_size = nodes.mailbox["features"].size(0)
            features = nodes.mailbox["features"].view(batch_size, -1)
            outputs = layer(features)
            return {"features": outputs}

        return layer_udf

    def _prepare_graph(self, graph):
        """
        Transform the input graph inplace into a graph that can be used for computation.

        To process all nodes simultaneously, we add a "padding" node that acts as a
        placeholder child or parent for every node missing one of those, so that every
        node has the same in-degree (4). The padding node always has a feature vector
        of zero.

        Edges are always indexed in the order (left child, right child, parent, self).

        The following invariants hold for all nodes:
            - Leaf nodes have 2 child edges from the padding node.
            - Unary function nodes have a left edge from their child and a right edge
              from the padding node.
            - Expression roots (nodes directly under Equality nodes) have a parent
              edge from the padding node, and there are no Equality nodes.

        Parameters
        ----------
        graph
            The graph to modify. Modified in place; if this is not desirable,
            clone the graph ahead of time and pass in the clone to modify in place.
        """
        # Remove Equality nodes, splitting examples into left and right expressions
        #root_mask = (
        #    graph.ndata[char_sp.FUNCTION_FIELD]
        #    == self.function_vocab[char_sp.EQUALITY_FUNCTION]
        #)
        #root_indices = torch.nonzero(root_mask, as_tuple=False)[:, 0]
        #graph.remove_nodes(root_indices)

        # Save the parent -> child edges for later
        parent_graph = graph.reverse()

        # Add a "padding node" to signal missing values
        graph.add_nodes(1)
        pad_node_id = graph.num_nodes() - 1
        graph.ndata["function"][pad_node_id] = self.function_vocab[
            char_sp.PAD_FUNCTION
        ]
        graph.ndata["token"][pad_node_id] = self.token_vocab[char_sp.PAD_TOKEN]

        # Grab node_ids of INT+ and INT- nodes
        int_mask = (graph.ndata["function"] == self.function_vocab['INT+']) | \
                    (graph.ndata["function"] == self.function_vocab['INT-'])

        # Add edges from the padding node to all nodes with less than 2 children
        pad_count = 2 - graph.in_degrees()  # Number of times to pad each node
        pad_count[int_mask] = 2 # Add 2 to pad count of integer nodes
        pad_count[pad_node_id] = 0  # Never add input edges to the pad node

        while not (pad_count <= 0).all():
            pad_mask = pad_count > 0
            pad_idx = torch.nonzero(pad_mask, as_tuple=False)[:, 0]
            graph.add_edges(pad_node_id, pad_idx)
            pad_count -= pad_mask.to(pad_count.dtype)

        # Add the parent -> child edges back in
        graph.add_edges(*parent_graph.edges())

        # Add edges from the pad node to every node without a parent, except itself
        pad_idx = torch.nonzero(graph.in_degrees() < 3, as_tuple=False)[:-1, 0]
        graph.add_edges(pad_node_id, pad_idx)

        # Add self-loops to all nodes except the pad node
        internal_indices = graph.nodes()[:-1]
        graph.add_edges(internal_indices, internal_indices)

        # Grab all the node_ids of the digits (don't want to message pass over this graph)
        int_ids = torch.nonzero(int_mask, as_tuple=True)[0]
        # Child id mask of INT+ and INT- nodes
        cid_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=graph.device)
        cid_mask.scatter_(0, graph.in_edges(int_ids)[0], 1)
        cid_mask &= (graph.ndata["function"] == self.function_vocab[char_sp.LEAF_FUNCTION])
        # Negate and get the nondigit node ids to apply message passing
        nondigit_ids = torch.nonzero(~cid_mask, as_tuple=True)[0]

        return nondigit_ids


    def _embed_nodes(self, graph):
        """
        Compute a feature vector for each node in the graph through embedding lookup.

        Each node is embedded based on its token (if it is a leaf) or function
        (if it is not).

        Parameters
        ----------
        graph
            The graph to compute features for.

        Returns
        -------
        Tensor
            A tensor of shape [num_nodes, self.hparams.d_model] containing per-node
            features.
        """
        leaf_mask = (
            graph.ndata[char_sp.FUNCTION_FIELD]
            == self.function_vocab[char_sp.LEAF_FUNCTION]
        )
        combined_idx = torch.where(
            leaf_mask,
            graph.ndata[char_sp.TOKEN_FIELD],
            graph.ndata[char_sp.FUNCTION_FIELD] + self.min_function_idx,
        )
        node_embeddings = self.embedding(combined_idx)
        return node_embeddings

    @staticmethod
    def _left_right_subgraphs(graph):
        """
        Given a graph containing multiple equations, produce subgraphs containing just
        the left (resp. the right) expressions in those equalities.

        Works with both a typical input graph and a computational graph processed
        through _prepare_graph().

        Parameters
        ----------
        graph
            A graph containing multiple equalities.

        Returns
        -------
        left_subgraph : DGLGraph
            A graph of expressions on the left-hand side of equations in the input.
        right_subgraph : DGLGraph
            A graph of expressions on the right-hand side of equations in the input.
        """
        left_subgraph = dgl.node_subgraph(
            graph, graph.ndata[char_sp.SIDE_FIELD] == char_sp.SIDE_LEFT
        )
        right_subgraph = dgl.node_subgraph(
            graph, graph.ndata[char_sp.SIDE_FIELD] == char_sp.SIDE_RIGHT
        )

        # Workaround for https://github.com/dmlc/dgl/issues/2310.
        # Equations list all of their left nodes before all of their right nodes, so
        # count the nodes on each side by running sum, then take alternating elements.
        subexpression_node_counts = torch.unique_consecutive(
            graph.ndata[char_sp.SIDE_FIELD], return_counts=True
        )[1]
        left_counts = subexpression_node_counts[::2]
        right_counts = subexpression_node_counts[1::2]

        left_subgraph.set_batch_num_nodes(left_counts)
        left_subgraph.set_batch_num_edges(left_counts - 1)

        right_subgraph.set_batch_num_nodes(right_counts)
        right_subgraph.set_batch_num_edges(right_counts - 1)

        return left_subgraph, right_subgraph


    # Currently does not work with multiple children
    def forward(self, graph, lengths, pad=False):
        """
        Compute the logits of equality holding for each equation in the (batched) graph.

        NOTE: modifies the graph in place.

        Parameters
        ----------
        graph
            The graph (possibly batched) containing equalities to classify.

        Returns
        -------
        Tensor
            For each equality in the input graph, the logit of positive classification.
        """
        nondigit_ids = self._prepare_graph(graph)
        graph.ndata["features"] = self._embed_nodes(graph)
        subgraph = graph.subgraph(nondigit_ids)
        for layer in self.layers:
            reduce_udf = self._reduce_layer(layer)
            subgraph.update_all(
                message_func=dgl.function.copy_src("features", "features"),
                reduce_func=reduce_udf,
            )
        pad_node_id = graph.num_nodes() - 1
        graph.remove_nodes(pad_node_id)
        #left_subgraph, right_subgraph = self._left_right_subgraphs(graph)

        # TODO: try other readouts

        lengths = lengths-2 if pad else lengths # Account for padding in the lengths

        unpadded_batch = torch.split(graph.ndata["features"], lengths.tolist()[::-1], dim=0)[::-1]
        batch = pad_tokens(self, unpadded_batch) if pad else unpadded_batch
        return pad_sequence(batch, padding_value=0.0, batch_first=True).squeeze(2)

