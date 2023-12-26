import unittest

class BuySellStock():
    def run(self, prices: list[int]) -> int:
        left = prices[0]
        right = prices[1]
        len_prices: int = len(prices)
        most_profit: int = 0

        for i in range(len_prices - 1):
            if left > right:
                left = prices[i]
                right = prices[i + 1]
            else:
                if right - left > most_profit:
                    most_profit = right - left
                right = prices[i + 1]
        return most_profit

class TestBuySellStock(unittest.TestCase):
    def setUp(self):
        self.buy_sell_stock = BuySellStock()
    
    def test_buy_sell_stock(self):
        self.assertEqual(self.buy_sell_stock.run([7,1,5,3,6,4]), 5)
        self.assertEqual(self.buy_sell_stock.run([7,6,4,3,1]), 0)

if __name__ == "__main__":
    unittest.main()

"""
121. Best Time to Buy and Sell Stock
Easy
29.4K
1K
Companies
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
 

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104
"""
