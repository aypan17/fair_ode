import dgl
import torch
import matplotlib.pyplot as plt
import networkx as nx

FUNCTION_FIELD = "function"  # The function at a (non-leaf) node
TOKEN_FIELD = "token"  # The token at a (leaf) node
SIDE_FIELD = "side"  # Whether a node is on the left, right, or root of an equation

LEAF_FUNCTION = "<Atom>"
EQUALITY_FUNCTION = "Equality"
PAD_FUNCTION = "<Pad_F>"

NONLEAF_TOKEN = "<Function>"
PAD_TOKEN = "<Pad_T>"

INDEX_PLACEHOLDER = -1

SIDE_ROOT = 0
SIDE_LEFT = 1
SIDE_RIGHT = 2

class Solution: 
    def _deserialize(self, equation, function_vocab, token_vocab):
        """
        Recursive helper function for `deserialize`. Nodes are traversed in reverse pre-order.

        Do not use this function directly; call `deserialize` instead.

        Parameters
        ----------
        equation
            Symbols in the tree's preorder traversal. Modified in place.
        function_vocab
            Mapping from function names to indices.
        token_vocab
            Mapping from token names to indices.

        Returns
        -------
        DGLGraph
            A homogeneous graph representing the parseable prefix of the serialized tree.
        """
        if not equation:
            raise ValueError(
                "Cannot deserialize an empty structure. Empty trees are serialized as '#'."
            )

        token = equation.pop(0)
        print(token)
        # Base case: non-integer leaf node
        if token in token_vocab:
            print(token)
            graph = dgl.graph(([], []))
            graph.add_nodes(1)
            graph.ndata[FUNCTION_FIELD] = torch.tensor([function_vocab[LEAF_FUNCTION]], dtype=torch.int64)
            graph.ndata[TOKEN_FIELD] = torch.tensor([token_vocab[token]], dtype=torch.int64)
            graph.ndata[SIDE_FIELD] = torch.tensor([SIDE_ROOT], dtype=torch.int64)

        # Base case: integer
        elif token in ['INT+', 'INT-']:
            digits = []
            while equation and (equation[0] in [str(i) for i in range(10)]):
                digits.append(equation.pop(0))

            graph = dgl.graph(([], []))
            graph.ndata[FUNCTION_FIELD] = torch.tensor([], dtype=torch.int64)
            graph.ndata[TOKEN_FIELD] = torch.tensor([], dtype=torch.int64)
            graph.ndata[SIDE_FIELD] = torch.tensor([], dtype=torch.int64)
            graph.add_nodes(len(digits)+1)

            graph.add_edges([i for i in range(len(digits))], len(digits)) # Connect INT node with each digit
            for i in range(len(digits)):
                graph.ndata[FUNCTION_FIELD][i] = function_vocab[LEAF_FUNCTION]
                # Reverse order to maintain reverse pre-order traversal
                graph.ndata[TOKEN_FIELD][i] = token_vocab[digits[len(digits)-i-1]]
                graph.ndata[SIDE_FIELD][i] = i+1

            graph.ndata[FUNCTION_FIELD][-1] = function_vocab[token]
            graph.ndata[TOKEN_FIELD][-1] = token_vocab[NONLEAF_TOKEN]
            graph.ndata[SIDE_FIELD][-1] = SIDE_ROOT
 
        # Operator
        else:
            function_id = function_vocab[token]
            token_id = token_vocab[NONLEAF_TOKEN]

            # Recursively collect left and right children
            if token in ['add','sub','mul','div','pow','rac']:
                left_subgraph = self._deserialize(equation, function_vocab, token_vocab)
            else:
                left_subgraph = dgl.graph(([], []))
            right_subgraph = self._deserialize(equation, function_vocab, token_vocab)

            right_root_idx = right_subgraph.num_nodes() - 1
            left_root_idx = right_root_idx + left_subgraph.num_nodes()
            root_idx = left_root_idx + 1

            if SIDE_FIELD in left_subgraph.ndata.keys():
                left_subgraph.ndata[SIDE_FIELD][:] = SIDE_LEFT
            if SIDE_FIELD in right_subgraph.ndata.keys():
                right_subgraph.ndata[SIDE_FIELD][:] = SIDE_RIGHT

            # Batch right child then left child. 
            # This enforces the node_ids to be labeled in reverse preorder traversal.
            graph = dgl.batch([right_subgraph, left_subgraph])
            graph.add_nodes(1)

            if left_subgraph.num_nodes() == 0:  # Right-unary
                graph.add_edges(right_root_idx, root_idx)
            elif right_subgraph.num_nodes() == 0:  # Left-unary
                graph.add_edges(left_root_idx, root_idx)
            else:  # Binary
                graph.add_edges([right_root_idx, left_root_idx], root_idx)

            graph.ndata[FUNCTION_FIELD][-1] = function_vocab[token]
            graph.ndata[TOKEN_FIELD][-1] = token_vocab[NONLEAF_TOKEN]
            graph.ndata[SIDE_FIELD][-1] = SIDE_ROOT

        return graph


    def deserialize(self, equation, f, t):
        """
        Deserialize binary equation data from a preorder token traversal
        into a DGLGraph. See the module docstring for the graph properties.

        Parameters
        ----------
        equation
            The "equation" field of a serialized example; a serialized binary tree.
        function_vocab
            A mapping from function name to id. See `build_vocabs()`.
        token_vocab
            A mapping from token name to id. See `build_vocabs()`.

        Returns
        -------
        DGLGraph
            The equation tree, deserialized into a usable data structure.
        """
        print(equation)
        return self._deserialize(equation, f, t)

def plot_graph(graph, function_vocab, token_vocab):
    """
    Plot an equation graph in a human-readable format.

    Each node is labeled with its token type if it is a leaf or its function type if it
    is internal, as well as its input index.

    Parameters
    ----------
    graph
        The graph to display.
    function_vocab
        A mapping from function name to id.
    token_vocab
        A mapping from token name to id.
    """
    function_names = {idx: name for name, idx in function_vocab.items()}
    token_names = {idx: name for name, idx in token_vocab.items()}

    labels = {}
    for i, (function_id, token_id) in enumerate(
        zip(graph.ndata[FUNCTION_FIELD].numpy(), graph.ndata[TOKEN_FIELD].numpy(),)
    ):
        if function_names[function_id] == LEAF_FUNCTION:
            labels[i] = f"{token_names[token_id]}"
        else:
            labels[i] = f"{function_names[function_id]}"
    print(labels)
    plt.subplots()
    nx_graph = graph.to_networkx()
    positions = nx.nx_agraph.graphviz_layout(nx_graph, prog="dot")
    nx.draw(nx_graph, positions)
    nx.draw_networkx_labels(nx_graph, positions, labels)
    plt.axis("off")
    #plt.draw()
    plt.show()
    plt.savefig("tester")
    #print("hi")
    #plt.ion()

def main():
    s = Solution()
    eq = ['sub', "Y'", 'add', 'x', 'atan', 'x']
    eq2 = ['sub', "Y'", 'mul', 'x', 'exp', 'mul', 'INT+', '4', '6', 'x']
    f = {'abs': 0, 'acos': 1, 'acosh': 2, 'acot': 3, 'acoth': 4, 'acsc': 5, 'acsch': 6, 'add': 7, 'asec': 8, 'asech': 9, 'asin': 10, 'asinh': 11, 'atan': 12, 'atanh': 13, 'cos': 14, 'cosh': 15, 'cot': 16, 'coth': 17, 'csc': 18, 'csch': 19, 'derivative': 20, 'div': 21, 'exp': 22, 'f': 23, 'g': 24, 'inv': 25, 'ln': 26, 'mul': 27, 'pow': 28, 'pow2': 29, 'pow3': 30, 'pow4': 31, 'pow5': 32, 'rac': 33, 'sec': 34, 'sech': 35, 'sign': 36, 'sin': 37, 'sinh': 38, 'sqrt': 39, 'sub': 40, 'tan': 41, 'tanh': 42, 'I': 43, 'INT': 44, 'INT+': 45, 'INT-': 46, '<Pad_F>': 47, '<Atom>': 48}
    t = {'<s>': 0, '</s>': 1, '<pad>': 2, '(': 3, ')': 4, '<SPECIAL_5>': 5, '<SPECIAL_6>': 6, '<SPECIAL_7>': 7, '<SPECIAL_8>': 8, '<SPECIAL_9>': 9, 'pi': 10, 'E': 11, 'x': 12, 'y': 13, 'z': 14, 't': 15, '0': 16, '1': 17, '2': 18, '3': 19, '4': 20, '5': 21, '6': 22, '7': 23, '8': 24, '9': 25, 'I': 26, 'Y': 37, "Y'": 38, "Y''": 39, '<Pad_T>': 40, '<Function>': 41}
    g = s.deserialize(eq, f, t)
    plot_graph(g, f, t)


if __name__ == '__main__':
    main()