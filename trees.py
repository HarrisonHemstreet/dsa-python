import unittest
from leetcode.nodes import TreeNode, tree1, tree2, tree3, tree4, tree5, tree6

class Tree():
    def bfs(self, root: TreeNode) -> list[int]:
        visited: list[TreeNode] = [root]
        res: list[int] = []

        while visited:
            curr_node = visited.pop(0)
            res.append(curr_node.val)
            if curr_node.left:
                visited.append(curr_node.left)
            if curr_node.right:
                visited.append(curr_node.right)
        return res

    def dfs_preorder(self, root: TreeNode) -> list[int]:
        res: list[int] = []
        def traverse(node: TreeNode):
            res.append(node.val)
            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
        traverse(root)
        return res

    def dfs_inorder(self, root: TreeNode) -> list[int]:
        res: list[int] = []
        def traverse(node: TreeNode):
            if node.left:
                traverse(node.left)
            res.append(node.val)
            if node.right:
                traverse(node.right)
        traverse(root)
        return res

    def dfs_postorder(self, root: TreeNode) -> list[int]:
        res: list[int] = []
        def traverse(node: TreeNode):
            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
            res.append(node.val)
        traverse(root)
        return res

class TestCode(unittest.TestCase):
    def setUp(self):
        self.trees = Tree()
    
    def test_bfs(self):
        self.assertEqual(self.trees.bfs(tree1), [1,2,3,4,5])
        self.assertEqual(self.trees.bfs(tree2), [5,3,8,1,4,9])
        self.assertEqual(self.trees.bfs(tree3), [1,2,3,4,5])
        self.assertEqual(self.trees.bfs(tree4), [6,4,3,5])
        self.assertEqual(self.trees.bfs(tree5), [8,12,10,14])
        self.assertEqual(self.trees.bfs(tree6), [15,13,18,11,20])
    
    def test_dfs_preorder(self):
        self.assertEqual(self.trees.dfs_preorder(tree1), [1,2,4,5,3])
        self.assertEqual(self.trees.dfs_preorder(tree2), [5,3,1,4,8,9])
        self.assertEqual(self.trees.dfs_preorder(tree3), [1,2,3,4,5])
        self.assertEqual(self.trees.dfs_preorder(tree4), [6,4,3,5])
        self.assertEqual(self.trees.dfs_preorder(tree5), [8,12,10,14])
        self.assertEqual(self.trees.dfs_preorder(tree6), [15,13,11,18,20])
    
    def test_dfs_inorder(self):
        self.assertEqual(self.trees.dfs_inorder(tree1), [4,2,5,1,3])
        self.assertEqual(self.trees.dfs_inorder(tree2), [1,3,4,5,8,9])
        self.assertEqual(self.trees.dfs_inorder(tree3), [2,1,4,3,5])
        self.assertEqual(self.trees.dfs_inorder(tree4), [3,4,5,6])
        self.assertEqual(self.trees.dfs_inorder(tree5), [8,10,12,14])
        self.assertEqual(self.trees.dfs_inorder(tree6), [11,13,15,18,20])

    def test_dfs_postorder(self):
        self.assertEqual(self.trees.dfs_postorder(tree1), [4,5,2,3,1])
        self.assertEqual(self.trees.dfs_postorder(tree2), [1,4,3,9,8,5])
        self.assertEqual(self.trees.dfs_postorder(tree3), [2,4,5,3,1])
        self.assertEqual(self.trees.dfs_postorder(tree4), [3,5,4,6])
        self.assertEqual(self.trees.dfs_postorder(tree5), [10,14,12,8])
        self.assertEqual(self.trees.dfs_postorder(tree6), [11,13,20,18,15])

if __name__ == "__main__":
    unittest.main()