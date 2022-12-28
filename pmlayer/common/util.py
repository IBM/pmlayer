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
