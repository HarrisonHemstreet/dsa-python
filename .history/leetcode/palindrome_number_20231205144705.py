from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        num_list: List[str] = list(f'{num}')
        rev_num: int = int("".join(num_list[::-1]))

        return False if num != rev_num

p = PalindromeNumber()
p.run(123)