class BTreeInOrderTraversal():
    def run(self, root: Optional[TreeNode]) -> List[int]:
        visited: List[int] = []
        current: TreeNode = root
        self.process_node(current, visited)
        return visited

    def process_node(self, node: Optional[TreeNode], visited: List[int]) -> List[int]:
        if not node:
            return visited
        if node and node.left is not None:
            self.process_node(node.left, visited)
        visited.append(node.val)
        if node and node.right is not None:
            self.process_node(node.right, visited)
        return visited

"""
94. Binary Tree Inorder Traversal
Easy
12.9K
705
Companies
Given the root of a binary tree, return the inorder traversal of its nodes' values.

 

Example 1:


Input: root = [1,null,2,3]
Output: [1,3,2]
Example 2:

Input: root = []
Output: []
Example 3:

Input: root = [1]
Output: [1]
 

Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100
 

Follow up: Recursive solution is trivial, could you do it iteratively?
"""