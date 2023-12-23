import unittest
from nodes import ListNode

class DeleteDups():
    def delete_dups(self, head: ListNode | None) -> ListNode | None:
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next

        return head

    def LL_to_list(self, node: ListNode) -> list[int]:
        node: ListNode = self.delete_dups(node)
        res: list[int] = []
        while node is not None:
            res.append(node.val)
            node = node.next
        return res

class TestDeleteDups(unittest.TestCase):
    def setUp(self):
        self.delete_dups = DeleteDups()

    def test_delete_dups(self):
        L1: ListNode = ListNode(1)
        L1_tail: ListNode = L1
        L1_tail.next = ListNode(1)
        L1_tail = L1_tail.next
        L1_tail.next = ListNode(2)
        L1_tail = L1_tail.next

        L1_res: ListNode = ListNode(1)
        L1_tail_res: ListNode = L1_res
        L1_tail_res.next = ListNode(2)
        L1_tail_res = L1_tail_res.next

        L2: ListNode = ListNode(1)
        L2_tail: ListNode = L2
        L2_tail.next = ListNode(1)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(1)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(2)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(3)
        L2_tail = L2_tail.next

        L2: ListNode = ListNode(1)
        L2_tail: ListNode = L2
        L2_tail.next = ListNode(2)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(3)
        L2_tail = L2_tail.next

        L3: ListNode = ListNode(1)
        L3_tail: ListNode = L3
        L3_tail.next = ListNode(1)
        L3_tail = L3_tail.next
        L3_tail.next = ListNode(1)
        L3_tail = L3_tail.next

        self.assertEqual(self.delete_dups.LL_to_list(L1), [1,2])
        self.assertEqual(self.delete_dups.LL_to_list(L2), [1,2,3])
        self.assertEqual(self.delete_dups.LL_to_list(L3), [1])

if __name__ == "__main__":
    unittest.main()

"""
83. Remove Duplicates from Sorted List
Easy
8.3K
277
Companies
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

 

Example 1:


Input: head = [1,1,2]
Output: [1,2]
Example 2:


Input: head = [1,1,2,3,3]
Output: [1,2,3]
 

Constraints:

The number of nodes in the list is in the range [0, 300].
-100 <= Node.val <= 100
The list is guaranteed to be sorted in ascending order.
"""