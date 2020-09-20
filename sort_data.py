import sys
import io

WORD2ID = {'<s>': 0, '</s>': 1, '<pad>': 2, '(': 3, ')': 4, '<SPECIAL_5>': 5, '<SPECIAL_6>': 6, '<SPECIAL_7>': 7, '<SPECIAL_8>': 8, '<SPECIAL_9>': 9, 'pi': 10, 'E': 11, 'x': 12, 'y': 13, 'z': 14, 't': 15, 'a0': 16, 'a1': 17, 'a2': 18, 'a3': 19, 'a4': 20, 'a5': 21, 'a6': 22, 'a7': 23, 'a8': 24, 'a9': 25, 'abs': 26, 'acos': 27, 'acosh': 28, 'acot': 29, 'acoth': 30, 'acsc': 31, 'acsch': 32, 'add': 33, 'asec': 34, 'asech': 35, 'asin': 36, 'asinh': 37, 'atan': 38, 'atanh': 39, 'cos': 40, 'cosh': 41, 'cot': 42, 'coth': 43, 'csc': 44, 'csch': 45, 'derivative': 46, 'div': 47, 'exp': 48, 'f': 49, 'g': 50, 'inv': 51, 'ln': 52, 'mul': 53, 'pow': 54, 'pow2': 55, 'pow3': 56, 'pow4': 57, 'pow5': 58, 'rac': 59, 'sec': 60, 'sech': 61, 'sign': 62, 'sin': 63, 'sinh': 64, 'sqrt': 65, 'sub': 66, 'tan': 67, 'tanh': 68, 'I': 69, 'INT+': 70, 'INT-': 71, 'INT': 72, 'FLOAT': 73, '-': 74, '.': 75, '10^': 76, 'Y': 77, "Y'": 78, "Y''": 79, '0': 80, '1': 81, '2': 82, '3': 83, '4': 84, '5': 85, '6': 86, '7': 87, '8': 88, '9': 89}
BIN_OPS = ['add', 'sub', 'mul', 'div', 'pow', 'rac', 'derivative', 'g']
UNA_OPS =  ['inv', 'pow2', 'pow3', 'pow4', 'pow5', 'sqrt', 'exp', 'ln', 'abs', 'sign', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch', 'f']


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
            raise ValueError("A tree can have the following children:" + "\n"
            "    lchild=None, rchild=None or" + "\n"
            "    lchild!=None, rchild=None or" + "\n"
            "    lchild!=None, rchild!=None or" + "\n"
            "Got the following instead:" + "\n"
            "    lchild=%s, rchild=%s" % (repr(lchild), repr(rchild)))
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


# Given a list of tokens in prefix form, convert to a BinaryEqnTree object for decoding.
def prefix_to_tree(tokens, raw, splits, counts):
    tree, _ = _prefix_to_tree(tokens)
    d = tree.get_depth()
    splits[d].append(raw)
    counts[d] += 1

def _prefix_to_tree(tokens, idx=0):
    token = tokens[idx]
    idx += 1
    if token in BIN_OPS:
        left, idx = _prefix_to_tree(tokens, idx=idx)
        right, idx = _prefix_to_tree(tokens, idx=idx) 
        root = BinaryEqnTree(token, left, right)
    elif token in UNA_OPS:
        left, idx = _prefix_to_tree(tokens, idx=idx)
        root = BinaryEqnTree(token, left, None)
    elif token in ['INT+', 'INT-']:
        val = ""
        while idx < len(tokens) and tokens[idx] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            val += tokens[idx]
            idx += 1
        root = BinaryEqnTree(token, None, None, value=int(val))
    else:
        root = BinaryEqnTree('Symbol', None, None, value=token)
    return root, idx

def main():
    inp = sys.argv[1]
    out = sys.argv[2]

    for suffix in ['test', 'valid', 'train']:
        with io.open(inp+'.'+suffix, mode='r', encoding='utf-8') as f1:
            # either reload the entire file, or the first N lines (for the training set)
            raw = [line for line in f1]
            lines = [line.rstrip().split('|') for line in raw]
            data = [xy.split('\t') for _, xy in lines]
            data = [xy for xy in data if len(xy) == 2]

        splits = [[] for _ in range(30)]
        counts = [0 for _ in range(30)]
        for xy, r in zip(data, raw):
            x, _ = xy
            x = x.split()
            prefix_to_tree(x, r, splits, counts)

        with io.open(out+'.'+suffix, mode='a', encoding='utf-8') as f2:
            for split in splits:
                if split:
                    for line in split:
                        f2.write(line)
        f1.close()
        f2.close()
        print(suffix)
        print(counts)


if __name__ == '__main__':
    main()