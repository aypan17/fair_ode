'''
class TreeLSTM_Encoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.emb_dim    
        self.dropout = params.dropout 
        self.cpu = params.cpu
        self.symmetric = params.symmetric # Whether or not to use symmetric blocks for add/mul
        self.max_order = params.order # Highest derivative that appears in the ode (currently 2)
        self.num_vars = params.vars # Number of variables in the ODE (currently 1)
        self.num_bit = params.num_bit # Max number of bits allowed in binary represenation

        # Symbol embeddings
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB), embedding_dim=self.dim))

        # Binary embeddings
        self.bin2emb = nn.Embedding(num_embeddings=self.num_bit, embedding_dim=self.dim)

        # Integer encoding block
        for sign in INT:
            setattr(self, sign, nn.Sequential(nn.Linear(self.dim, self.dim), nn.Tanh(), nn.Linear(self.dim, self.dim)))

        # EOS encoding block
        setattr(self, '#', UnaryLSTMNode(self.dim, self.dim))

        # Derivative operators
        for d in DIFFERENTIALS:
            setattr(self, d, UnaryLSTMNode(self.dim, self.dim))

        # Unary operators
        for op in UNARY:
            setattr(self, op, UnaryLSTMNode(self.dim, self.dim))

        # Binary operators
        for op in BINARY:
            if self.symmetric and (op == 'add' or op == 'mul'):
                setattr(self, op, BinaryLSTMNodeSym(num_input=self.dim, num_hidden=self.dim))
            else:
                setattr(self, op, BinaryLSTMNode(num_input=self.dim, num_hidden=self.dim))

        # Not sure how to add bias in a sensible manner
        # self.bias = nn.Parameter(torch.FloatTensor([0]))

        
    # We don't use attention yet, so casual is set to False
    def forward(self, x=None, causal=False):
        batch, id2emb, lengths, invalid = self.prefix_to_tree(x)
        children, parents, embeddings, buckets, max_depth, batch_size, roots = self.label_and_map_batch(batch)

        # Compute nodes
        for depth in range(1, max_depth + 1):
            input_ids = buckets[depth]
            inputs, ids = self.concatenate_inputs(self.dim, input_ids, parents, embeddings)

            for key in inputs:
                nn_block = getattr(self, key)
                if key == SYMBOL_ENCODER or key in INT:
                    hidden, cell = inputs[key]
                    output = (nn_block(hidden), cell)
                elif key == EOS or key in UNARY or key in DIFFERENTIALS:
                    hidden, cell = inputs[key]
                    output = nn_block((hidden, cell), dropout=self.dropout)
                elif key in BINARY:
                    output = nn_block(inputs[key][0], inputs[key][1], dropout=self.dropout)
                else:
                    raise AssertionError("The given key is not valid. Key:", key)

                # For a given key, set the embedding for each id in ids[key] in sequential order.
                for j in range(len(ids[key])):
                    embeddings[ids[key][j]] = (output[0][j], output[1][j])

        max_len = torch.max(lengths).item()
        output = self.flatten_tree(embeddings, id2emb, max_len)
        assert list(output.size()) == [batch_size, max_len, self.dim]
        return output, lengths, invalid

    # Helper function for concatenating inputs for models with a hidden state and cell state. 
    # For fixed depth, takes a dictionary of idx of child nodes and their embeddings.
    # Returns a concatenated tuple of (hidden state, cell state) and a dictionary of parent ids, indexed by key.
    def concatenate_inputs(self, dim, input_ids, parents, embeddings):
        inputs = {}
        ids = {}
        for key in input_ids:
            if key == SYMBOL_ENCODER or key in INT: 
                children = input_ids[key][0]
                hidden_state = torch.stack([embeddings[child_id] for child_id in children], dim=0)
                cell_state = torch.zeros((len(children), dim), dtype=torch.float)
                cell_state = cell_state if self.cpu else cell_state.cuda()
                inputs[key] = hidden_state, cell_state
            elif key == EOS or key in UNARY or key in DIFFERENTIALS:
                children = input_ids[key][0]
                hidden_state = torch.stack([embeddings[child_id][0] for child_id in children], dim=0)
                cell_state = torch.stack([embeddings[child_id][1] for child_id in children], dim=0)
                inputs[key] = hidden_state, cell_state
            elif key in BINARY:
                lchildren = input_ids[key][0]
                rchildren = input_ids[key][1]
                assert(len(lchildren) == len(rchildren) and len(lchildren) > 0)
                lhide = torch.stack([embeddings[child_id][0] for child_id in lchildren], dim=0)
                lcell = torch.stack([embeddings[child_id][1] for child_id in lchildren], dim=0)
                rhide = torch.stack([embeddings[child_id][0] for child_id in rchildren], dim=0)
                rcell = torch.stack([embeddings[child_id][1] for child_id in rchildren], dim=0)
                inputs[key] = [(lhide, lcell), (rhide, rcell)]
            else:
                AssertionError("[%s] is not a valid block", key)
            ids[key] = [parents[idx] for idx in input_ids[key][0]]
        return inputs, ids

    # Given batch of binary trees with computed embeddings, returns tensor of size [maxlen, batch_size, model_dim] by flattening each tree
    def flatten_tree(self, embeddings, id2emb, maxlen):        
        id2emb = [tree + [tree[0]] + [0] * (maxlen-1-len(tree)) for tree in id2emb] # Use maxlen-1 because we already account for the end padding
        zeros = torch.zeros(self.dim, dtype=torch.float)
        zeros = zeros if self.cpu else zeros.cuda()
        embeddings[0] = zeros, 0
        return torch.stack([torch.stack([embeddings[idx][0] for idx in tree], dim=0) for tree in id2emb], dim=0)

    # Returns the embedding of an integer using its binary embedding
    def int_emb(self, integer):
        b = bin(integer)[2:]
        if len(b) > self.num_bit:
            return "Number too large"
        place = torch.LongTensor([i for i in range(len(b)) if b[len(b)-i-1] == '1'])
        place = place if self.cpu else place.cuda()
        return torch.sum(self.bin2emb(place), 0)

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

    # Given a list of tokens in prefix form, convert to a BinaryEqnTree object for decoding.
    def prefix_to_tree(self, token_list):
        trees = []
        id2embs = []
        lengths = []
        valid_idx = []
        counter = 1
        for i in range(len(token_list)):
            old_counter = counter # reset the counter in case we come across an invalid equation
            tree, id2emb, _, counter, valid = self._prefix_to_tree(token_list[i], [], counter=counter)
            if valid:
                trees.append(tree)
                id2embs.append(id2emb)
                lengths.append(len(id2emb)+1) # Add 1 to account for the end padding
                valid_idx.append(i)
            else:
                counter = old_counter
        return trees, id2embs, torch.LongTensor(lengths), valid_idx

    def _prefix_to_tree(self, tokens, id2emb, idx=0, counter=1):
        token = tokens[idx]
        idx += 1
        id2emb.append(counter)
        counter += 1
        valid = True

        if token in BINARY:
            left, _, idx, counter, valid1 = self._prefix_to_tree(tokens, id2emb, idx=idx, counter=counter)
            right, _, idx, counter, valid2 = self._prefix_to_tree(tokens, id2emb, idx=idx, counter=counter) 
            valid = valid1 and valid2
            root = BinaryEqnTree(token, left, right)
        elif token == EOS or token in UNARY:
            left, _, idx, counter, valid = self._prefix_to_tree(tokens, id2emb, idx=idx, counter=counter)
            root = BinaryEqnTree(token, left, None)
        elif token in INT:
            counter += 1
            val = ""
            while idx < len(tokens) and tokens[idx] in DIGITS:
                val += tokens[idx]
                idx += 1
            if int(val) >= (1 << self.num_bit):
                valid = False
            root = BinaryEqnTree(token, BinaryEqnTree(val, None, None, value=self.int_emb(int(val))), None)
        elif token in DERIVATIVES:
            counter += 2
            value = torch.LongTensor([VOCAB['Y']])[0]
            value = value if self.cpu else value.cuda()
            root = BinaryEqnTree(SYMBOL_ENCODER, BinaryEqnTree('Y', None, None, value=value), None)
            root = BinaryEqnTree('d'+str(len(token)-1), root, None)
        elif token in LEAF:
            counter += 1
            value = torch.LongTensor([VOCAB[token]])[0]
            value = value if self.cpu else value.cuda()
            root = BinaryEqnTree(SYMBOL_ENCODER, BinaryEqnTree(token, None, None, value=value), None)
        else:
            raise AssertionError("{0} is not a valid symbol".format(token))

        root.depth = root.get_depth()
        return root, id2emb, idx, counter, valid


class BinaryLSTMNodeSym(torch.nn.Module):

    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=False)

        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_by_self = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_by_opposite = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, input_left, input_right, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = input_left
        hr, cr = input_right
        i = torch.sigmoid(self.data(hl) + self.data(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_by_self(hl) +
                           self.forget_by_opposite(
                               hr) + self.forget_bias)
        f_right = torch.sigmoid(self.forget_by_opposite(hl) +
                           self.forget_by_self(
                               hr) + self.forget_bias)
        o = torch.sigmoid(self.output(hl) + self.output(hr) + self.output_bias)
        u = torch.tanh(self.input(hl) + self.input(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        return h, c


class BinaryLSTMNode(torch.nn.Module):

    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data_left = nn.Linear(num_input, num_hidden, bias=False)
        self.data_right = nn.Linear(num_input, num_hidden, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_left_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_left_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output_left = nn.Linear(num_input, num_hidden, bias=False)
        self.output_right = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input_left = nn.Linear(num_input, num_hidden, bias=False)
        self.input_right = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, input_left, input_right, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = input_left
        hr, cr = input_right
        i = torch.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_left_by_left(hl) +
                           self.forget_left_by_right(
                               hr) + self.forget_bias_left)
        f_right = torch.sigmoid(self.forget_right_by_left(hl) +
                           self.forget_right_by_right(
                               hr) + self.forget_bias_right)
        o = torch.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = torch.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        return h, c


class UnaryLSTMNode(torch.nn.Module):
    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=True)
        self.forget = nn.Linear(num_input, num_hidden, bias=True)
        self.output = nn.Linear(num_input, num_hidden, bias=True)
        self.input = nn.Linear(num_input, num_hidden, bias=True)

    def forward(self, inp, dropout=None):
        """

        Args:
            inp: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        h, c = inp
        i = torch.sigmoid(self.data(h))
        f = torch.sigmoid(self.forget(h))
        o = torch.sigmoid(self.output(h))
        u = torch.tanh(self.input(h))
        if dropout is None:
            c = i * u + f * c
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f * c
        h = o * torch.tanh(c)
        return h, c
'''