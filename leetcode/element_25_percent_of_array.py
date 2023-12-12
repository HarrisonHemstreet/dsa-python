from collections import Counter
import unittest

class TwentyFivePercentOfArray():
    def run(self, arr: list[int]) -> int:
        return Counter(arr).most_common(1)[0][0]

class TestCode(unittest.TestCase):
    def setUp(self):
        self.twenty_five_percent = TwentyFivePercentOfArray()

    def test_twenty_five_percent(self):
        self.assertEqual(self.twenty_five_percent.run([1,2,2,6,6,6,6,7,10]), 6)
        self.assertEqual(self.twenty_five_percent.run([1,1]), 1)

if __name__ == "__main__":
    unittest.main()
"""
1287. Element Appearing More Than 25% In Sorted Array
Easy
1.6K
72
Companies
Given an integer array sorted in non-decreasing order, there is exactly one integer in the array that occurs more than 25% of the time, return that integer.

 

Example 1:

Input: arr = [1,2,2,6,6,6,6,7,10]
Output: 6
Example 2:

Input: arr = [1,1]
Output: 1
 

Constraints:

1 <= arr.length <= 104
0 <= arr[i] <= 105
"""
