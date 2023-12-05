from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        l_n: List[int] = list(f'{num}')
        print("l_n:", l_n)

p = PalindromeNumber()
p.run(121)