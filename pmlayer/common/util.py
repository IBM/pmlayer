def generate_indices(monotonicity, length):
    idx_inc = []
    idx_dec = []
    idx_none = []
    if monotonicity > 0:
        idx_inc = list(range(length))
    elif monotonicity < 0:
        idx_dec = list(range(length))
    else:
        idx_none = list(range(length))
    return (idx_inc, idx_dec, idx_none)

def parse_monotonicity(monotonicity):
    if monotonicity is None:
        return 0
    elif isinstance(monotonicity, int):
        if -1 <= monotonicity <= 1:
            return monotonicity
    elif isinstance(monotonicity, str):
        if monotonicity == 'increasing':
            return 1
        elif monotonicity == 'decreasing':
            return -1
        elif monotonicity == 'none':
            return 0
    raise ValueError('Unknown monotonicity value: %s' % monotonicity)

def parse_monotonicities(monotonicities, length):
    if isinstance(monotonicities, list):
        idx_inc = []
        idx_dec = []
        idx_none = []
        for i, m in enumerate(monotonicities):
            monotonicity = parse_monotonicity(m)
            if monotonicity > 0:
                idx_inc.append(i)
            elif monotonicity < 0:
                idx_dec.append(i)
            else:
                idx_none.append(i)
        return (idx_inc, idx_dec, idx_none)

    monotonicity = parse_monotonicity(monotonicities)
    return generate_indices(monotonicity, length)

class TreeNode:
    def __init__(self, left, value, right):
        self.left = left
        self.value = value
        self.right = right
        self.left_node = None
        self.right_node = None

def traverse_preorder(node):
    ret = [ (node.left, node.value, node.right) ]
    if node.left_node is not None:
        ret.extend(traverse_preorder(node.left_node))
    if node.right_node is not None:
        ret.extend(traverse_preorder(node.right_node))
    return ret

def create_skewed_tree(max_val):
    root = TreeNode(0, 1, max_val)
    bt = root
    for i in range(2, max_val):
        bt.right_node = TreeNode(i-1, i, max_val)
        bt = bt.right_node
    return root
