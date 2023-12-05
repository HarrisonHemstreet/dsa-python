from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        l_n: List[str] = list(f'{num}')
        r_n: List[str] = l_n
        r_n.reverse()
        print("l_n:", l_n)
        print("r_n:", r_n)

p = PalindromeNumber()
p.run(121)