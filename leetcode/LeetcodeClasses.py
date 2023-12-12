class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
    1
   / \
  2   3
 / \
4   5
"""
tree1 = TreeNode(1)
tree1.left = TreeNode(2)
tree1.right = TreeNode(3)
tree1.left.left = TreeNode(4)
tree1.left.right = TreeNode(5)

"""
    5
   / \
  3   8
 / \   \
1   4   9
"""
tree2 = TreeNode(5)
tree2.left = TreeNode(3)
tree2.right = TreeNode(8)
tree2.left.left = TreeNode(1)
tree2.left.right = TreeNode(4)
tree2.right.right = TreeNode(9)

"""
  1
  / \
 2   3
    / \
   4   5
"""
tree3 = TreeNode(1)
tree3.left = TreeNode(2)
tree3.right = TreeNode(3)
tree3.right.left = TreeNode(4)
tree3.right.right = TreeNode(5)

"""
    6
   / 
  4   
 / \
3   5
"""
tree4 = TreeNode(6)
tree4.left = TreeNode(4)
tree4.left.left = TreeNode(3)
tree4.left.right = TreeNode(5)

"""
  8
   \
    12
   /  \
  10  14
"""
tree5 = TreeNode(8)
tree5.right = TreeNode(12)
tree5.right.left = TreeNode(10)
tree5.right.right = TreeNode(14)

"""
   15
   / \
  13  18
 /     \
11      20
"""
tree6 = TreeNode(15)
tree6.left = TreeNode(13)
tree6.right = TreeNode(18)
tree6.left.left = TreeNode(11)
tree6.right.right = TreeNode(20)

"""
    1
   / 
  1
"""
tree7 = TreeNode(1)
tree7.left = TreeNode(1)

"""
 1
  \
   1
"""
tree8 = TreeNode(1)
tree8.left = TreeNode(None)
tree8.right = TreeNode(1)

# My modified Leetcode version:
# class ListNode:
#     def __init__(self, val = 0, next = None):
#         self.val: int | None = val
#         self.next: self = next

#     def __repr__(self):
#         return "ListNode(val=" + str(self.val) + ", next={" + str(self.next) + "})"
    
#     def __eq__(self, other):
#         if isinstance(other, ListNode):
#             current_self = self
#             current_other = other
#             while current_self is not None and current_other is not None:
#                 if current_self.val != current_other.val:
#                     return False
#                 current_self = current_self.next
#                 current_other = current_other.next
#             return current_self is None and current_other is None
#         return False

# print(ListNode(1, ListNode(4, ListNode(5, None))))

# Leetcode's Definition for singly-linked list.