import unittest
from LeetcodeClasses import ListNode

class DeleteDups():
    def run(self, head: ListNode | None) -> ListNode | None:
        head_pointer = head
        head2 = head
        accu: set[int] = set()
        # accu.add(head.val)
        while head.next is not None:
            print("head.val:", head.val)
            if head.val not in accu:
                accu.add(head.val)
                # continue
            head = head.next

        while head2.next is not None:
            print("head2.val:", head2.val)
            head2 = head.val

        return head_pointer

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

        self.assertEqual(self.delete_dups.run(L1), L1_res)
        self.assertEqual(self.delete_dups.run(L2), L2_res)

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