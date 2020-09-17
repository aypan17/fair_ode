import argparse
import math
import bisect
import torch
import torch.nn as nn
from torch.autograd import grad
from collections import defaultdict

from src.envs.char_sp import EOS, LEAF, VOCAB, BINARY, UNARY, DERIVATIVES, DIFFERENTIALS, INT, DIGITS

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

words = DIGITS + LEAF + BINARY + UNARY + INT + DIFFERENTIALS + [EOS] 
id2word = {i: s for i, s in enumerate(words)}
word2id = {s: i for i, s in id2word.items()}


class BinaryLSTM(torch.nn.Module):

    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=False)
        self.data_left = nn.Linear(num_input, num_hidden, bias=False)
        self.data_right = nn.Linear(num_input, num_hidden, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_left_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_left_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output = nn.Linear(num_input, num_hidden, bias=False)
        self.output_left = nn.Linear(num_input, num_hidden, bias=False)
        self.output_right = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input = nn.Linear(num_input, num_hidden, bias=False)
        self.input_left = nn.Linear(num_input, num_hidden, bias=False)
        self.input_right = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, hl, cl, hr, cr, p, dropout=None):
        """

        Args:
            input: (lhide, lcell), (rhide, rcell), phide

        Returns:
            (num_hidden,), (num_hidden)
        """
        #hl, cl, hr, cr, p = input
        print(hl.size())
        print(cl.size())
        print(hr.size())
        print(cr.size())
        print(p.size())
        i = torch.sigmoid(self.data(p) + self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget(p) + self.forget_left_by_left(hl) +
                           self.forget_left_by_right(hr) + self.forget_bias_left)
        f_right = torch.sigmoid(self.forget(p) + self.forget_right_by_left(hl) +
                           self.forget_right_by_right(hr) + self.forget_bias_right)
        o = torch.sigmoid(self.output(p) + self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = torch.tanh(self.input(p) + self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        return h, c


class UnaryLSTM(torch.nn.Module):
    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=False)
        self.child_data = nn.Linear(num_input, num_hidden, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget = nn.Linear(num_input, num_hidden, bias=False)
        self.child_forget = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output = nn.Linear(num_input, num_hidden, bias=False)
        self.child_output = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input = nn.Linear(num_input, num_hidden, bias=False)
        self.child_input = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, inp, dropout=None):
        """

        Args:
            inp: (child_hidden, child_cell), (parent_hidden)

        Returns:
            (bs, num_hidden), (bs, num_hidden)
        """
        h, c, p = inp
        i = torch.sigmoid(self.data(p) + self.child_data(h) + self.data_bias)
        f = torch.sigmoid(self.forget(p) + self.child_forget(h) + self.forget_bias)
        o = torch.sigmoid(self.output(p) + self.child_output(h) + self.output_bias)
        u = torch.tanh(self.input(p) + self.child_input(h) + self.input_bias)
        if dropout is None:
            c = i * u + f * c
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f * c
        h = o * torch.tanh(c)
        return h, c


class Eqn:

    emb = None
    dim = None
    cpu = True
    id2w = id2word
    w2id = word2id
    pad_id = len(id2w)

    def __init__(self, ids, model):
        self.nodes = []
        self.arr = [0] * 256
        self.n2w = {}
        self.emb = getattr(model, 'emb')
        self.dim = getattr(model, 'dim')
        self.cpu = getattr(model, 'cpu')
        self.build(ids)

    def build(self, ids):
        self._build(ids[0])
        self.nodes = sorted(self.nodes)

    def _build(self, ids, idx=0, node=0):
        wid = ids[idx].item()
        word = self.id2w[wid]
        value = torch.LongTensor([wid])[0]
        value = value if self.cpu else value.cuda()
        self.arr[node] = self.emb(value)
        self.n2w[node] = word
        self.nodes.append(node)

        pad = torch.LongTensor([self.pad_id])[0]
        pad = pad if self.cpu else pad.cuda()
        zeros = self.emb(pad)
        idx += 1
        if word in BINARY:
            idx = self._build(ids, idx=idx, node=2*node+1)
            idx = self._build(ids, idx=idx, node=2*node+2) 
        elif word == EOS or word in UNARY:
            idx = self._build(ids, idx=idx, node=2*node+1)
            self.arr[2*node+2] = zeros
            self.nodes.add(2*node+2)
            self.n2w[2*node+2] = 'PAD'
        elif word in INT:
            val = ''
            while idx < len(ids) and self.id2w[ids[idx].item()] in DIGITS:
                val += self.id2w[ids[idx].item()]
                idx += 1
            val = int(val)
            self.arr[2*node+1] = self.int_emb(val)
            self.arr[2*node+2] = zeros
            self.nodes.append(2*node+1)
            self.nodes.append(2*node+2)
            self.n2w[2*node+1] = val
            self.n2w[2*node+2] = 'PAD'
        elif word in DIFFERENTIALS:
            yvalue = torch.LongTensor([self.w2id['Y']])[0]
            yvalue = value if self.cpu else value.cuda()
            yvalue = self.emb(yvalue)
            self.arr[2*node+1] = yvalue
            self.arr[2*node+2] = zeros
            self.nodes.append(2*node+1)
            self.nodes.append(2*node+2)
            self.n2w[2*node+1] = yvalue
            self.n2w[2*node+2] = 'PAD'
        elif word in LEAF:
            pass
        else:
            raise AssertionError("{0} is not a valid symbol".format(word))
        return idx

    # Returns the embedding of an integer using its binary embedding
    def int_emb(self, integer):
        b = bin(integer)[2:]
        place = torch.LongTensor([i for i in range(len(b)) if b[len(b)-i-1] == '1'])
        place = place if self.cpu else place.cuda()
        place = torch.sum(self.emb(place), 0)
        return place

    def get_depth(self):
        return 1+int(math.log2(self.nodes[len(self.nodes)-1]+1))

    def get_emb(self):
        emb = []
        for node in self.nodes:
            emb.append(self.arr[node])
        return emb

    def get_eqn(self):
        words = []
        for node in self.nodes:
            words.append(self.n2w[node])
        return words

    def get_nodes(self):
        return self.nodes

    def preprocess(self):
        depth = self.get_depth()
        order = []
        lo = 0
        for d in range(1, depth+1):
            i = bisect.bisect_left(self.nodes,2**d-1,lo=lo)
            order.append(self.nodes[lo:i])
            lo = i 
        order = order[1:]
        order.reverse()
        return self.get_emb(), order, {idx:pos for pos, idx in enumerate(self.nodes)}

    '''
    def _update(self, node_union):
        zeros = Eqn.emb(Eqn.pad_id)
        for node in node_union - set(self.nodes):
            self.arr[node] = zeros
            self.nodes.add(node)
            self.n2w[node] = 'PAD'

    @classmethod
    def build_midpoints(cls, inp, base, steps=30):
        node_union = union(set(inp.get_nodes()), set(base.get_nodes()))
        inp._update(node_union)
        base._update(node_union)
    '''



class BinaryEqnTree:

    NULL = "#"

    def __init__(self, function_name, lchild, rchild, value=None,
                 is_a_floating_point=False, raw=None, label=None, depth=None):
        """

        Args:
            function_name: the name of the node
            lchild: the left child (a BinaryEqnTree or None)
            rchild: the right child (a BinaryEqnTree or None)
        """
        #TODO: make value a more general construct, i.e. a dictionary, or an object so that more than one value can be stored at a node
        if lchild is None and rchild is not None:
            raise ValueError("Bad tree")
        self.function_name = function_name
        self.lchild = lchild
        self.rchild = rchild
        self.is_a_floating_point = is_a_floating_point
        self.value = value
        self.is_binary = lchild is not None and rchild is not None
        self.is_unary = lchild is not None and rchild is None
        self.is_leaf = lchild is None and rchild is None
        self.raw = raw
        self.label = label
        self.depth = depth
        self.cls = None

    def apply(self, fn):
        if self.lchild is not None:
            self.lchild.apply(fn)
        if self.rchild is not None:
            self.rchild.apply(fn)
        fn(self)


    def get_depth(self):
        left = 0
        right = 0
        if self.lchild:
            left = self.lchild.get_depth()
        if self.rchild:
            right = self.rchild.get_depth()
        return 1 + max(left, right)

    """
    Runs a DFS to label all nodes and create the children, embedding dictionaries.

    Args: 
        BinaryEqnTree: tree in the batch to be labeled and embedded.

        unused_id: one-element list containing lowest value unused id in the batch. Need list for mutability.
                    Updated with every call.

        current_depth: integer equal to the depth of the node calling the 

        children: a Dict<id, list<id>> whose key is the id of a node and value is the list of its children ids.
                    Updated with every call.

        parent: a Dict<id, id> whose key is the id of a child and value is the id of its parent.
                    Update with every call.

        embedding: a Dict<id, embedding_vector> whose key is the id of a node in the tree
                    and value is the embedding vector of the node; a node w/o embedding has value None.
                    Updated with every call.

        buckets: a list of defaultDict<function_name, [list<lchild_id>, list<rchild_id>]> indexed by depth. 
                    Each defaultDict has function_name keys and [list<lchild_id>, list<rchild_id>] values 
                    that contain the lchild and rchild ids of the [function_name] block.
                    Updated with every call.

    Returns:
        idx: the integer id of the node that calls labels_embeddings.
    """
    def label_and_map_tree(self, unused_id, current_depth, children, parent, embedding, buckets):
        idx = unused_id[0]
        unused_id[0] += 1
        self.depth = current_depth
        current_depth -= 1
        children[idx] = []
        embedding[idx] = self.value
        if self.lchild:
            lchild_id = self.lchild.label_and_map_tree(unused_id, current_depth, children, parent, embedding, buckets)
            children[idx].append(lchild_id)
            buckets[self.depth][self.function_name][0].append(lchild_id)
        if self.rchild:
            rchild_id = self.rchild.label_and_map_tree(unused_id, current_depth, children, parent, embedding, buckets)
            children[idx].append(rchild_id)
            buckets[self.depth][self.function_name][1].append(rchild_id)
        if (not self.lchild) and (not self.rchild):
            buckets[self.depth][self.function_name][0].append(idx)
        for child_id in children[idx]:
            parent[child_id] = idx
        return idx

    def is_numeric(self):
        if self.function_name != "eq":
            print("Warning: is_numeric should only be called on the root of an equation tree")
            return False #raise ValueError("is_numeric should only be called on the root of an equation tree")
        return self._is_numeric()

    def _is_numeric(self):
        if self.is_leaf:
            return self.is_a_floating_point
        if self.is_unary:
            return self.lchild._is_numeric()
        if self.is_binary:
            return self.lchild._is_numeric() or self.rchild._is_numeric()
        raise AssertionError(str(self))

    def __str__(self):
        if self.is_binary:
            return "{}({}, {})".format(self.function_name,
                                       str(self.lchild),
                                       str(self.rchild))
        elif self.is_unary:
            return "{}({})".format(self.function_name,
                                   str(self.lchild))
        elif self.is_leaf:
            return "{}={}".format(self.function_name, self.value)
        else:
            raise RuntimeError("Invalid tree:\n%s" % repr(self))


class TreeLSTM_Verifier(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.emb_dim    
        self.dropout = params.dropout 
        self.cpu = params.cpu
        self.symmetric = params.symmetric # Whether or not to use symmetric blocks for add/mul
        self.max_order = params.order # Highest derivative that appears in the ode (currently 2)
        self.num_vars = params.vars # Number of variables in the ODE (currently 1)
        self.num_bit = params.num_bit # Max number of bits allowed in binary represenation

        # Embeddings
        self.pad = len(id2word)
        self.emb = nn.Embedding(num_embeddings=self.pad+1, embedding_dim=self.dim, padding_idx=self.pad)
        #self.embedding = self.embedding if self.cpu else self.embedding.cuda()

        # NN blocks
        self.b = BinaryLSTM(self.dim, self.dim)
        #self.u = UnaryLSTM(self.dim, self.dim)

        self.logits = torch.nn.Softmax(dim=1)
        self.bias = nn.Parameter(torch.FloatTensor([0]))

    # Emb: list of embeddings in DFS order 
    # order: computation order of the DFS (first row is leaves)
    # n2p: dict; key = node in order, value = pos in emb
    def forward(self, inp):
        e = Eqn(inp, self)
        emb, order, n2p = e.preprocess()
        lstm = {}
        for level in order:
            lhide = []
            lcell = []
            rhide = []
            rcell = []
            state = []
            parents = []
            for node in level:
                lchild = lstm.get(2*node+1)
                rchild = lstm.get(2*node+2)
                if not lchild:
                    lstm[node] = emb[n2p[node]], torch.zeros(self.dim, dtype=torch.float)
                else:
                    lhide.append(lchild[0])
                    lcell.append(lchild[1])
                    rhide.append(rchild[0])
                    rcell.append(rchild[1])
                    state.append(emb[n2p[node]])
                    parents.append(node)
            if parents:
                pstate, pcell = self.b(torch.stack(lhide), torch.stack(lcell), torch.stack(rhide), torch.stack(rcell), torch.stack(state))
                for i in range(len(parents)):
                    lstm[parents[i]] = pstate[i], pcell[i]
        print("done")
        return torch.dot(lstm[1][0], lstm[2][0])


    # Returns the embedding of an integer using its binary embedding
    def int_emb(self, integer):
        b = bin(integer)[2:]
        if len(b) > self.num_bit:
            return "Number too large"
        place = torch.LongTensor([word2id[str(i)] for i in range(len(b)) if b[len(b)-i-1] == '1'])
        place = place if self.cpu else place.cuda()
        place = torch.sum(self.emb(place), 0)
        return place

    # Helper function for labeling batch. Calls label_and_map_tree() for the BinEqnTree object.
    def label_and_map_batch(self, batch):
        max_depth = max([tree.depth for tree in batch]) + 1 # Add 1 to account for additional leaves
        unused_id = [1]
        root = []
        children = {}
        parents = {}
        embedding = {}
        def child_ids():
            return [[], []]
        buckets = [defaultdict(child_ids) for i in range(max_depth + 1)]
        labels = []
        depths = []
        batch_size = 0
        for tree in batch:
            current_depth = tree.depth + 1
            tree_children = {}
            tree_parents = {}
            tree_embedding = {}
            root.append(unused_id[0])
            tree.label_and_map_tree(unused_id, current_depth, tree_children, tree_parents, tree_embedding, buckets)
            children.update(tree_children)
            parents.update(tree_parents)
            embedding.update(tree_embedding)      
            batch_size += 1
        return children, parents, embedding, buckets, max_depth, batch_size, root

    # Given a list of ids in prefix form, convert to a BinaryEqnTree object for classification.
    def id_to_tree(self, id_list):
        trees = []
        embs = []
        lengths = []
        valid_idx = []
        counter = 1
        for i in range(len(id_list)):
            word_list = [id2word[idx.item()] for idx in id_list[i] if idx.item() != self.pad]
            tree, emb, _ = self._prefix_to_tree(word_list, [])
            trees.append(tree)
            embs.extend(emb)    
        return trees, embs

    def _prefix_to_tree(self, words, emb, idx=0):
        word = words[idx]
        idx += 1
        value = torch.LongTensor([word2id[word]])[0]
        value = value if self.cpu else value.cuda()
        value = self.emb(value)
        zeros = torch.zeros(self.emb, dtype=torch.float)
        zeros = zeros if self.cpu else zeros.cuda()
        z_tree = BinaryEqnTree('PAD', None, None, values=zeros)
        emb.append(value)
        if word in BINARY:
            left, emb, idx = self._prefix_to_tree(words, emb, idx=idx)
            right, emb, idx = self._prefix_to_tree(words, emb, idx=idx) 
            root = BinaryEqnTree(word, left, right, value=value)
        elif word == EOS or word in UNARY:
            left, _, idx = self._prefix_to_tree(words, emb, idx=idx)
            emb.append(zeros)
            root = BinaryEqnTree(word, left, z_tree, value=value)
        elif word in INT:
            val = ''
            while idx < len(words) and words[idx] in DIGITS:
                val += words[idx]
                idx += 1
            val = self.int_emb(int(val))
            zeros = torch.zeros(self.emb, dtype=torch.float, device='cpu' if self.cpu else 'cuda:0')
            emb.append(val)
            emb.append()
            root = BinaryEqnTree(word, BinaryEqnTree('INT', None, None, value=val), z_tree, value=value)
        elif word in DIFFERENTIALS:
            yvalue = torch.LongTensor([word2id['Y']])[0]
            yvalue = value if self.cpu else value.cuda()
            yvalue = self.emb(yvalue)
            emb.append(yvalue)
            root = BinaryEqnTree('Y', None, None, value=yvalue)
            root = BinaryEqnTree(word, root, z_tree, value=value)
        elif word in LEAF:
            root = BinaryEqnTree(word, None, None, value=value)
        else:
            raise AssertionError("{0} is not a valid symbol".format(word))

        root.depth = root.get_depth()
        return root, emb, idx
    '''
    # We don't use attention yet, so casual is set to False
    def forward(self, x, ordercausal=False):
        print(x)
        batch, emb = self.id_to_tree(x)
        children, parents, embeddings, buckets, max_depth, batch_size, root = self.label_and_map_batch(batch)
        lstm_state = {}
        print(children)
        # Compute nodes
        for depth in range(1, max_depth + 1):
            bucket = buckets[depth]
            inp_emb, inp_ids = self.concatenate_inputs(self.dim, bucket, parents, embeddings, lstm_state)
            for key in inp_emb:
                if key in BINARY:
                    out = self.b(inp_emb[key])
                else:
                    #print(key)
                    #print(inp_emb[key][0].size())
                    #print(inp_emb[key][1].size())
                    #print(inp_emb[key][2].size())
                    out = self.u(inp_emb[key])
                # For a given key, set the embedding for each id in ids[key] in sequential order.
                for j in range(len(inp_ids[key])):
                    lstm_state[inp_ids[key][j]] = (out[0][j], out[1][j])

        assert len(root) == batch_size
        
        lhs = torch.stack([lstm_state[children[idx][0]][0] for idx in root], dim=0)
        rhs = torch.stack([lstm_state[children[idx][1]][0] for idx in root], dim=0)
        dim = lhs.size()
        print('dim')
        print(dim)
        batched_dot = torch.bmm(lhs.view(dim[0], 1, dim[1]), rhs.view(dim[0], dim[1], 1)).add(self.bias)
        #out = torch.cat((torch.zeros((batch_size, 1)), batched_dot.squeeze(2)), dim=1)
        #logits = self.logits(out)
        print(batched_dot)
        return batched_dot, emb #torch.index_select()
    '''
    # Helper function for concatenating inputs for models with a hidden state and cell state. 
    # For fixed depth, takes a dictionary of idx of child nodes and their embeddings.
    # Returns a concatenated tuple of (hidden state, cell state) and a dictionary of parent ids, indexed by key.
    def concatenate_inputs(self, dim, bucket, parents, embeddings, lstm_state):
        inp_emb = {}
        inp_ids = {}
        for key in bucket:
            p = [parents[idx] for idx in bucket[key][0]]
            p_state = torch.stack([embeddings[idx] for idx in p], dim=0)
            if key in LEAF or key in ['INT', 'Y']:
                ids = bucket[key][0]
                state = torch.stack([embeddings[idx] for idx in ids], dim=0)
                pad = torch.zeros(state.size(), dtype=torch.float, requires_grad=True)
                print("get inputs", key)
                print(state.size())
                print(pad.size())
                inp_emb[key] = pad, pad, state
                inp_ids[key] = ids
            elif key == EOS or key in UNARY or key in INT or key in DIFFERENTIALS:
                children = bucket[key][0]
                hidden = torch.stack([lstm_state[idx][0] for idx in children], dim=0)
                cell = torch.stack([lstm_state[idx][1] for idx in children], dim=0)
                print("get inputs", key)
                print(hidden.size())
                print(p_state.size())
                inp_emb[key] = hidden, cell, p_state
                inp_ids[key] = p
            elif key in BINARY:
                lchildren = bucket[key][0]
                rchildren = bucket[key][1]
                lhide = torch.stack([lstm_state[idx][0] for idx in lchildren], dim=0)
                lcell = torch.stack([lstm_state[idx][1] for idx in lchildren], dim=0)
                rhide = torch.stack([lstm_state[idx][0] for idx in rchildren], dim=0)
                rcell = torch.stack([lstm_state[idx][1] for idx in rchildren], dim=0)
                inp_emb[key] = lhide, lcell, rhide, rcell, p_state
                inp_ids[key] = p
            elif key == 'eq':
                pass
            else:
                AssertionError("[%s] is not a valid block", key)
        return inp_emb, inp_ids

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=8, help='hidden layer size')
    parser.add_argument('--dropout', type=float, default=0, help='layer dropout')
    parser.add_argument('--cpu', type=bool, default=True, help='use cpu' )
    parser.add_argument('--symmetric', type=bool, default=False, help='use symmetric LSTM nodes for add/mul')
    parser.add_argument('--order', type=int, default=2, help='max order of ODE')
    parser.add_argument('--vars', type=int, default=1, help='max number of variables in ODE')
    parser.add_argument('--num_bit', type=int, default=10, help='max bits used to represent number')
    params = parser.parse_args()
    return params

def pad_ids(inp, base, w2id):
    assert len(inp) == len(base)
    inp = [[w2id[w] for w in e.split()] for e in inp]
    base = [[w2id[w] for w in e.split()] for e in base]
    for i in range(len(inp)):
        l = max(len(inp[i]), len(base[i]))
        inp[i] += [len(w2id)] * (l - len(inp[i]))
        inp[i] += [len(w2id)] * (l - len(base[i]))
    return torch.LongTensor(inp), torch.LongTensor(base)

def main():

    #inp = ["eq sub x INT+ 3 sin add x INT+ 3", "eq add x INT+ 2 5 add x INT+ 2 2"]
    #base = ["eq sin add x INT+ 3 sin add x INT+ 3", "eq add x INT+ 2 2 add x INT+ 2 2"]
    inp = ["eq add x INT+ 2 5 add x INT+ 2 2"]
    base = ["eq add x INT+ 2 2 add x INT+ 2 2"]
    inp, base = pad_ids(inp, base, word2id)
    model = TreeLSTM_Verifier(get_params())
    model.eval()
    #e = Eqn(model=model)
    #i = Eqn(model=model, ids=inp)
    #b = Eqn(model=model, ids=base)
    #inp, order, n2p = i.preprocess()
    #base = b.get_emb()
    #batch, _, _, _ = model.id_to_tree(inp_ids)
    #interpretable_embedding = configure_interpretable_embedding_layer(model, 'emb')
    #inp_emb = interpretable_embedding.indices_to_embeddings(inp_ids)
    #base_emb = interpretable_embedding.indices_to_embeddings(base_ids)
    lig = LayerIntegratedGradients(model, model.emb)
    attributions = lig.attribute(inp, base, n_steps=23)
    #print(grad(outputs=out, inputs=emb, allow_unused=True))
    #print(grad(outputs=out2, inputs=emb2, allow_unused=True))

if __name__ == '__main__':
    main()