# contains a head, tail and length property.
# each node points to another node or null.
class Node:
    def __init__(self, val, next = None, prev = None):
        self.val: int = val
        self.next: Node | None = next
        self.prev: Node | None = prev

    def __str__(self):
        return (f"""
            Node(
                val: {self.val}, next: {self.next}
            )
        """)
        # return (f"""
        #     Node(
        #         val: {self.val}, prev: {self.prev}
        #     )
        # """)

class DoublyLinkedList:
    def __init__(self, head = None, tail = None, length = 0):
        self.head: Node | None = head
        self.tail: Node | None = tail
        self.length: int = length

    def __str__(self):
        return f"DoublyLinkedList next: {self.head}"
        # return f"DoublyLinkedList prev: {self.tail}"

    # O(1)
    def push(self, val: int):
        new_node = Node(val)

        if self.head == None:
            self.head = new_node
            self.tail = new_node
        elif self.tail is not None:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.length += 1
        return self

    def get_len(self):
        return self.length

    # O(n)
    def find(self, val: int):
        curr_node = self.head
        while curr_node is not None:
            if curr_node.val == val:
                return curr_node
            else:
                curr_node = curr_node.next
        if curr_node == None:
            return f"Could not find {val}"

    # O(1) / O(n)
    def pop(self):
        curr_node = self.head
        while curr_node.next.next is not None:
            curr_node = curr_node.next
        curr_node.next = None
        self.tail = curr_node
        self.length -= 1
        return self.head

    # O(1)
    def shift(self, val: int):
        if self.head == None:
            return "LIST IS EMPTY!"
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node
        self.length += 1
        return self.head

    # O(1)
    def unshift(self):
        if self.head == None:
            return "LIST IS EMPTY!"
        self.head = self.head.next
        self.length -= 1
        return self.head

    # take in an "index" and return the node located there
    def get(self, index):
        curr_node = self.head
        i = index
        while i > 1:
            i -= 1
            curr_node = curr_node.next
        return curr_node

    # take in value and an index of the node we are changing the value to
    def set(self, val, index):
        curr_node = self.head
        i = index
        while i > 1:
            i -= 1
            curr_node = curr_node.next
        curr_node.val = val
        return curr_node
    
    # like set, but inserts a new node instead of changing the value
    def insert(self, val, index):
        if index < 0 or index > self.length:
            return False
        new_node = Node(val)
        curr_node = self.head
        i = index
        while i > 2:
            i -= 1
            curr_node = curr_node.next
        prev_node = curr_node
        next_node = curr_node.next
        new_node.prev = prev_node
        new_node.next = next_node
        print("new_node:", new_node)
        curr_node = new_node

        # curr_node.next = new_node
        # curr_node.next.next = node_after
        # curr_node.prev = node_prev
        self.length += 1
        return self

    # like insert but the opposite
    def remove(self, index):
        curr_node = self.head
        i = index
        while i > 2:
            i -= 1
            curr_node = curr_node.next
        to_remove = curr_node.next
        curr_node.next = curr_node.next.next
        return to_remove

    def reverse(self):
        if self.head is None or self.tail is None:
            return self
        
        prev = None
        curr = self.head

        while curr is not None:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next

        accu = prev
        self.head = accu
        return prev

d_list = DoublyLinkedList()
d_list.push(1)
d_list.push(2)
d_list.push(3)
d_list.push(4)
d_list.push(5)
d_list.push(6)
d_list.push(7)
d_list.push(8)
print("d_list.head", d_list.head)
print("d_list.tail", d_list.tail)
print("d_list.length", d_list.length)
print("find:", d_list.find(8))
print("popped:", d_list.pop())
print("find:", d_list.find(8))
print("shift:", d_list.shift(0))
print("unshift:", d_list.unshift())
print(d_list.get_len())
print("get:", d_list.get(4))
print("set:", d_list.set(100, 4))
print("insert:", d_list.insert(200, 5))
# print("remove:", s_list.remove(5))
# print("d_list:", d_list.print_list())
print("d_list:", d_list)
# print("s_list reverse:", s_list.reverse())
