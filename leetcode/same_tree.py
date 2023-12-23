import unittest
from nodes import TreeNode, tree1, tree2, tree3, tree7, tree8

class SameTree():
    def run(self, p: TreeNode | None, q: TreeNode | None) -> bool:
        if p is None and q is None:
            return True
        elif p is None or q is None:
            return False
        if p.val != q.val:
            return False 
        return self.run(p.left, q.left) and self.run( p.right, q.right)

class TestSameTree(unittest.TestCase):
    def setUp(self):
        self.same_tree = SameTree()

    def test_same_tree(self):
        self.assertEqual(self.same_tree.run(tree1, tree1), True)
        self.assertEqual(self.same_tree.run(tree1, tree2), False)
        self.assertEqual(self.same_tree.run(tree2, tree3), False)
        self.assertEqual(self.same_tree.run(tree7, tree8), False)

if __name__ == "__main__":
    unittest.main()

"""
100. Same Tree
Easy
10.7K
214
Companies
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

 

Example 1:


Input: p = [1,2,3], q = [1,2,3]
Output: true
Example 2:


Input: p = [1,2], q = [1,null,2]
Output: false
Example 3:


Input: p = [1,2,1], q = [1,1,2]
Output: false
 

Constraints:

The number of nodes in both trees is in the range [0, 100].
-104 <= Node.val <= 104
"""