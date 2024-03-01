import unittest

class MaxOddBinaryNumber:
    def run(self, s: str) -> str:
        res = ""
        one_count = s.count("1")
        l_s = sorted(s)
        if one_count == 0:
            return s
        if one_count > 0:
            for i in range(len(s) - 1, -1, -1):
                if i < len(s) - 1:
                    res += l_s[i]
            return res + "1"
        return res

class TestCode(unittest.TestCase):
    def setUp(self):
        self.max_odd = MaxOddBinaryNumber()
    def test_max_odd(self):
        self.assertEqual(self.max_odd.run("100"), "001")
        self.assertEqual(self.max_odd.run("010"), "001")
        self.assertEqual(self.max_odd.run("0101"), "1001")

if __name__ == "__main__":
    unittest.main()
