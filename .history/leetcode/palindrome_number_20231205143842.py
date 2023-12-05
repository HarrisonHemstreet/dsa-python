from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        l_n: List[str] = list(f'{num}')
        r_n: List[str] = l_n
        print("l_n:", l_n)

p = PalindromeNumber()
p.run(121)