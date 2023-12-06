from typing import Optional, List
from Leetcode import ListNode
import unittest

class MergeLists:
    def run(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # print("list1:", list1.val < list2.val)
        # print("list2:", list2.val)
        # look at both the current nodes of list1, list2 and compare, adding the smallest of the two to the res ListNode. return head
        head = ListNode()
        curr = head

        while list1.next and list2.next:
            if list1.val < list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
    
        curr.next = list1 if list1 else list2
        print("head.next:", head.next)

        return head.next

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.merge_lists = MergeLists()

    def test_merge_lists(self):
        # self.assertEqual(self.merge_lists.run(
        #     ListNode(1, ListNode(2, ListNode(4, None))),
        #     ListNode(1, ListNode(3, ListNode(4, None)))
        # ),
        # ListNode(1, ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(4, None)))))))
        self.assertEqual(self.merge_lists.run(ListNode(None, None), ListNode(None, None)), ListNode(None, None))
        self.assertEqual(self.merge_lists.run(ListNode(None, None), ListNode(0, None)), ListNode(0, None))
    
    def test_merge_two_sorted_lists(self):
        list1 = ListNode(1, ListNode(2, ListNode(4)))
        list2 = ListNode(1, ListNode(3, ListNode(4)))
        expected = ListNode(1, ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(4))))))
        result = self.merge_lists.run(list1, list2)
        assert result == expected, f"Expected {expected}, but got {result}"

if __name__ == '__main__':
    unittest.main()

"""
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

 

Example 1:


Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: list1 = [], list2 = []
Output: []
Example 3:

Input: list1 = [], list2 = [0]
Output: [0]
 

Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both list1 and list2 are sorted in non-decreasing order.
"""