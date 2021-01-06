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


class UnaryLSTM(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.data = nn.Linear(d_model, d_model, bias=True)
        self.forget = nn.Linear(d_model, d_model, bias=True)
        self.output = nn.Linear(d_model, d_model, bias=True)
        self.input = nn.Linear(d_model, d_model, bias=True)

    def forward(self, inp: torch.Tensor, dropout=None) -> torch.Tensor:
        h, c = torch.split(inp, 1, dim=1)
        h = h.squeeze(1)
        c = c.squeeze(1)
        i = torch.sigmoid(self.data(h))
        f = torch.sigmoid(self.forget(h))
        o = torch.sigmoid(self.output(h))
        u = torch.tanh(self.input(h))
        if dropout is None:
            c = i * u + f * c
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f * c
        h = o * torch.tanh(c)
        h_norm = h / h.norm(p=2)
        c_norm = c / c.norm(p=2)
        return torch.stack([h_norm, c_norm], dim=1)


class BinaryLSTM(torch.nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.data_left = nn.Linear(d_model, d_model, bias=False)
        self.data_right = nn.Linear(d_model, d_model, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_left_by_left = nn.Linear(d_model, d_model, bias=False)
        self.forget_left_by_right = nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_left = nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_right = nn.Linear(d_model, d_model, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.output_left = nn.Linear(d_model, d_model, bias=False)
        self.output_right = nn.Linear(d_model, d_model, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.input_left = nn.Linear(d_model, d_model, bias=False)
        self.input_right = nn.Linear(d_model, d_model, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * d_model))

    def forward(self, inp_left, inp_right, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = torch.split(inp_left, 1, dim=1)
        hr, cr = torch.split(inp_right, 1, dim=1)
        hl = hl.squeeze(1)
        cl = cl.squeeze(1)
        hr = hr.squeeze(1)
        cr = cr.squeeze(1)
        i = torch.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_left_by_left(hl) + self.forget_left_by_right(hr) + self.forget_bias_left)
        f_right = torch.sigmoid(self.forget_right_by_left(hl) + self.forget_right_by_right(hr) + self.forget_bias_right)
        o = torch.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = torch.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        h_norm = h / h.norm(p=2)
        c_norm = c / c.norm(p=2)
        return torch.stack([h_norm, c_norm], dim=1)


class TreeLSTM_Encoders(torch.nn.Module):
    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__()

        self.d_model = params.emb_dim
        self.id2word = id2word
        self.word2id = word2id
        self.una_ops = una_ops
        self.bin_ops = bin_ops
        self.pad_idx = params.pad_index
        self.unary_modules = torch.nn.ModuleDict(
            {f: UnaryLSTM(params.emb_dim) for f in una_ops}
        )
        self.binary_modules = torch.nn.ModuleDict(
            {f: BinaryLSTM(params.emb_dim) for f in bin_ops}
        )
        self.leaf_emb = torch.nn.Embedding(
            num_embeddings=len(id2word),
            embedding_dim=params.emb_dim,
            padding_idx=self.pad_idx,
            max_norm=1.0,
        )
        self.num_enc = torch.nn.Sequential(
            nn.Linear(1, params.emb_dim),
            nn.Sigmoid(),
            nn.Linear(params.emb_dim, params.emb_dim),
            nn.Sigmoid()
        )
        if params.character_rnn:
            self.ch_rnn = torch.nn.RNN(params.emb_dim, params.emb_dim, dropout=params.dropout)

    '''
    operations: torch.Tensor,
    tokens: torch.Tensor,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    depths: torch.Tensor,
    operation_order: torch.Tensor,
    digits: torch.Tensor,
    integers: torch.Tensor
    '''
    def forward(self, x=None, lengths=None, augment=False, seq_num=False):
        if augment:
            return self.forward_augment(x)
        return self.forward_(x, lengths, seq_num)

    def forward_(
        self, x, lengths, seq_num
    ) -> torch.Tensor:
        """
        Given a batch of tensorized trees, encode the trees and return the hidden state of each node.
        """
        operations, tokens, left_idx, right_idx, depths, operation_order, _, integers, int_lens = x
        num_steps = operation_order.numel()
        num_nodes = operations.numel()
        activations = torch.zeros(
            (num_nodes, 2, self.d_model), dtype=torch.float, device=operations.device
        )

        if seq_num:
            int_emb = torch.zeros(
                (num_nodes, max(int_lens), self.d_model), dtype=torch.float, device=operations.device
            )
            int_mask = torch.zeros(num_nodes, dtype=torch.bool, device=operations.device)

        s = time.time()
        for depth in range(num_steps):  # type: ignore
            step_mask = depths == depth  # Indices to compute at this step
            op = operation_order[depth].item()

            if op in [-1, -2]: # Embedding lookup or number encoding
                idx = tokens.masked_select(step_mask)
                step_activations = self.leaf_emb(idx) if op == -1 else self.num_enc(idx.float().unsqueeze(1))
                zeros = torch.zeros(
                    (len(idx), self.d_model), dtype=torch.float, device=operations.device
                )
                step_activations = torch.stack([step_activations, zeros], dim=1)

                if op == -2 and seq_num:
                    int_mask = step_mask
                    int_emb = int_emb.masked_scatter(
                        int_mask.unsqueeze(1).unsqueeze(1), self.rnn(integers, int_lens)
                    )

            else:
                op_name = self.id2word[op]

                if op_name in self.unary_modules.keys():
                    module = self.unary_modules[op_name]
                    child_input_idx = left_idx.masked_select(step_mask)
                    child_activations = activations[child_input_idx]
                    step_activations = module(child_activations)
                else:  # Binary; equality operations always have a depth of -1
                    module = self.binary_modules[op_name]
                    left_input_idx = left_idx.masked_select(step_mask)
                    left_activations = activations[left_input_idx]
                    right_input_idx = right_idx.masked_select(step_mask)
                    right_activations = activations[right_input_idx]
                    step_activations = module(left_activations, right_activations)

            # Write computed activations into the shared buffer; NOTE: one copy of this
            # buffer is computed for each step, to allow for dense backprop
            activations = activations.masked_scatter(
                torch.stack([step_mask, step_mask], dim=1).unsqueeze(2), step_activations
            )
        e = time.time()
        print("Forward:", e-s)
        hidden, _ = torch.split(activations, 1, dim=1)

        if seq_num:
            s = time.time()
            zeros = torch.zeros(
                (num_nodes, max(int_lens)-1, self.d_model), dtype=torch.float, device=hidden.device
            )
            hidden_pad = torch.cat([hidden, zeros], dim=1)
            hidden_pad = torch.where(int_mask.unsqueeze(1).unsqueeze(1), int_emb, hidden_pad)
            dim = hidden_pad.size()
            hidden_pad_flat = hidden_pad.view(dim[0]*dim[1], self.d_model)
            nonzero = (hidden_pad_flat != 0).all(1)
            hidden = hidden_pad_flat[nonzero, :]
            e = time.time()
            print("Replace emb:", e-s)

        unpadded_batch = torch.split(hidden, lengths.tolist(), dim=0)
        return pad_sequence(unpadded_batch, padding_value=0.0, batch_first=True).squeeze(2)

    '''
    Computes a batch of equations with the form Y' - k*exp(x), Y=k*exp(x) for k in integers. 
    Augments the training for the NUMBER_ENCODER block.
    '''
    def forward_augment(self, x):
        # Intialize cell states
        zeros = torch.zeros((len(x), 1, self.d_model), dtype=torch.float, device=x.device)

        # Y' and x and EOS embeddings
        eos = self.word2id['<s>'] * torch.ones((len(x), 1), dtype=torch.long, device=x.device)
        y_token = self.word2id["Y'"] * torch.ones((len(x), 1), dtype=torch.long, device=x.device)
        x_token = self.word2id['x'] * torch.ones((len(x), 1), dtype=torch.long, device=x.device)
        eos, y_token, x_token = torch.split(self.leaf_emb(torch.cat([y_token, x_token, eos], dim=1)), 1, dim=1)

        # exp(x) embedding
        exp_module = self.unary_modules['exp']
        exp = exp_module(torch.cat([x_token, zeros], dim=1))

        # number embedding
        int_emb = self.num_enc(x.unsqueeze(1)).unsqueeze(1)

        # mul embedding
        mul_module = self.binary_modules['mul']
        mul = mul_module(torch.cat([int_emb, zeros], dim=1), exp)

        # sub embedding
        sub_module = self.binary_modules['sub']
        sub = sub_module(torch.cat([x_token, zeros], dim=1), mul)

        # split into hidden, cell
        exp, _ = torch.split(exp, 1, dim=1) 
        mul, _ = torch.split(mul, 1, dim=1)
        sub, _ = torch.split(sub, 1, dim=1)

        return torch.cat([eos, sub, y_token, mul, int_emb, exp, x_token, eos], dim=1)

    def rnn(self, integers, int_lens):
        s = time.time()
        emb = self.leaf_emb(integers)
        packed_inp = pack_padded_sequence(emb, int_lens.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_out, cell = self.ch_rnn(packed_inp)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        e = time.time()
        print("RNN emb:", e-s)
        return output
