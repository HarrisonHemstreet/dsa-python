import unittest
from two_sum import TwoSum
from palindrome_number import PalindromeNumber
from roman_to_int import RomanToInt

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.two_sum = TwoSum()
        self.palindrome_number = PalindromeNumber()
        self.roman_to_int = RomanToInt()

    def test_two_sum(self):
        self.assertEqual(self.two_sum.run([2,7,11,15], 9), [0,1])
        self.assertEqual(self.two_sum.run([3,2,4], 6), [1,2])
        self.assertEqual(self.two_sum.run([3,3], 6), [0,1])
    
    def test_palindrome_number(self):
        self.assertEqual(self.palindrome_number.run(121), True)
        self.assertEqual(self.palindrome_number.run(-121), False)
        self.assertEqual(self.palindrome_number.run(123), False)
    
    def test_roman_to_int(self):
        self.assertEqual(self.roman_to_int.run("III"), 3)
        self.assertEqual(self.roman_to_int.run(""))

if __name__ == '__main__':
    unittest.main()
