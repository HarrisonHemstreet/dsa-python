from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        num_list: List[str] = list(f'{num}')
        rev_num: int = int("".join(num_list[::-1]))

        if num == rev_num:
            return True
        else:
            return False

p = PalindromeNumber()
p.run(123)