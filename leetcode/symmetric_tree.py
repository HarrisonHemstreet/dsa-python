from nodes import TreeNode, tree9, tree10
import unittest

class SymmetricTree():
    def run(self, root: TreeNode | None) -> bool:
        if not root:
            return False
        return self.is_same(root.left, root.right)

    def is_same(self, left: TreeNode, right: TreeNode) -> bool:
        if left == None and right == None:
            return True
        if left == None or right == None:
            return False
        if left.val != right.val:
            return False
        return self.is_same(left.left, right.right) and self.is_same(right.left, left.right)

class TestSymmetricTree(unittest.TestCase):
    def setUp(self):
        self.symmetric_tree = SymmetricTree()

    def test_symmetric_tree(self):
        self.assertEqual(self.symmetric_tree.run(tree9), True)
        self.assertEqual(self.symmetric_tree.run(tree10), False)

if __name__ == "__main__":
    unittest.main()

"""
101. Symmetric Tree
Easy
14.7K
351
Companies
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

Example 1:


Input: root = [1,2,2,3,4,4,3]
Output: true
Example 2:


Input: root = [1,2,2,null,3,null,3]
Output: false
 

Constraints:

The number of nodes in the tree is in the range [1, 1000].
-100 <= Node.val <= 100
 

Follow up: Could you solve it both recursively and iteratively?
"""