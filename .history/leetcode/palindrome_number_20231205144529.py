from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        num_list: List[str] = list(f'{num}')
        rev_num: int = int(l_n[::-1])

        print("l_n:", num_list)
        print("r_n:", r_n)

p = PalindromeNumber()
p.run(123)