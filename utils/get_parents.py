import json
import sys
import io
import numpy as np

def get_parents(left, right):
        """
        Get id of parent nodes given left and right child ids.

        Parameters
        ----------
        left
            A tensor of indicies corresponding to the left child of each node, 
            or -1 if the node does not have a left child.
        right
            A tensor of indicies corresponding to the right child of each node, 
            or -1 if the node does not have a right child.

        Returns
        -------
        parents
            A tensor of indicies corresponding to the parent node of each node,
            or -1 if the node does not have a parent (the root node).
        """
        left = left
        right = right
        parents = [-1] * len(left)
        for idx in range(len(left)): 
            if left[idx] != -1:
                parents[left[idx]] = idx
            if right[idx] != -1:
                parents[right[idx]] = idx
        return parents

def main():
    left = sys.argv[1]
    right = sys.argv[2]
    with io.open(left, mode='r', encoding='utf-8') as left:
        l_idx = [np.array(json.loads(line)) for line in left]
    with io.open(right, mode='r', encoding='utf-8') as right:
        r_idx = [np.array(json.loads(line)) for line in right]
    file_handler_parents = io.open(sys.argv[3], mode='a', encoding='utf-8')

    for (l, r) in zip(l_idx, r_idx):
        p = get_parents(l, r)
        file_handler_parents.write(f'{p}\n')
        file_handler_parents.flush()

if __name__ == '__main__':
    main()