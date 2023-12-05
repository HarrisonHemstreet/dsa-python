import unittest
from two_sum import TwoSum
from palindrome_number import PalindromeNumber

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.two_sum = TwoSum()
        self.palindrome_number = PalindromeNumber()

    def test_two_sum(self):
        self.assertEqual(self.two_sum.run([2,7,11,15], 9), [0,1])
        self.assertEqual(self.two_sum.run([3,2,4], 6), [1,2])
        self.assertEqual(self.two_sum.run([3,3], 6), [0,1])
    
    def test_palindrome_number(self):
        self.assertEqual(self.palindrome_number.run(121))
        self.assertEqual(self.)

if __name__ == '__main__':
    unittest.main()
