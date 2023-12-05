from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        num_list: List[str] = list(f'{num}')
        rev_num: int = int(num_list[::-1])

        print("l_n:", num_list)
        print("r_n:", rev_num)

p = PalindromeNumber()
p.run(123)