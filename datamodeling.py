
#1. Array
class ArrayModel:
    def __init__(self, element, list_name):
        self.element = element
        self.list_name = list_name
        self.lists = {}  

    def add_to_list(self):
        if self.list_name not in self.lists:
            self.lists[self.list_name] = []
        self.lists[self.list_name].append(self.element)

    def get_list(self):
        return self.lists.get(self.list_name, [])

model1 = ArrayModel(5, "my_list")
model1.add_to_list()
print(model1.get_list())  

#2. Linked List_1
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        #node -> (data + pointer)

class LinkedList:
    def __init__(self, *args): #args == "가변인자"(여기서 args == "list"형식으로 보면은 되겠군)
        if not args:
            raise ValueError("least one element is required")
        
        self.head = Node(args[0])
        current = self.head #current = self.head(현재 node == "head")

        for data in args[1:]:
            current.next = Node(data)
            current = current.next

        self.tail = current #current == pointer.
        """마지막 node를 list의 끝으로 나타내는 tail로 설정."""

    def display_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("none")

#2-1 Linked List_2 (원형 연결 list)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList1:
    def __init__(self, *args):
        if not args:
            raise ValueError("least one element is required")
        
        self.head = Node(args[0])
        current = self.head

        for data in args[1:]:
            current.next = Node(data)
            current = current.next

            self.tail = current
            self.tail.next = self.head

    def display_list(self):
        current = self.head
        while True :
            print(current.data, end=" -> ")
            current = current.next
            if current == self.head:
                break
        print("Back to Head ->", current.data)
        
#2-2 Linked List_3(이중 연결 list)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.before = None 

class LinkedList2:
    def __init__(self, *args):
        if not args:
            raise ValueError("least one element is required")
        
        self.head = Node(args[0])#head 설정 완료.
        current = self.head

        for data in args[1:]:
            new_node = Node(data)
            current.next = new_node #현재 node next를 새 노드로 연결.
            new_node.before = current #새 노드의 이전을 현재 node로 연결.
            current = new_node #그 후 current를 새 노드로 갱신.

        self.tail = current

    def display_list_forward(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next

        print("None")
    
    def display_list_backward(self):
        current = self.tail
        while current:
            print(current.data, end=" -> ")
            current = current.before
        print("None")

#3. Stack
class Stacker:

    def __init__(self, name, *args):
        self.name = name
        self.lists = {}

        if not args:
            raise ValueError("nothing imported")
        
        self.lists[self.name] = list(args)
        
    def pushing(self, value):
        self.lists[self.name].append(value)
        
    def pop(self):
        if self.lists[self.name]:
            return self.lists[self.name].pop()
        else:
            raise IndexError("empty stack")

    def peek_top(self):
        if self.lists[self.name]:
            return self.lists[self.name]
        else:
            raise IndexError("empty stack")

#4. Queue
class Queue:

    def __init__(self, name, args):
        self.name = name
        self.lists = {}

        if not args:
            raise ValueError("nothing imported")
        
        self.lists[self.name] = list(args)

    def enqueque(self, value):
        self.lists[self.name].append(value)

    def Dequeue(self):
        if self.lists[self.name]:
            return self.lists[self.name].pop(0)
        else:
            raise IndexError("empty")

    def show(self):
        if self.lists[self.name]:
            return self.lists[self.name]
        else:
            raise IndexError("empty queue")
#4-1. Linear Queue
class Queue1:

    def __init__(self, name, args):
        self.name = name
        n = len(args)
        mylist = list(args) + [None] * 3

        self.lists = {}

        if not args:
            raise ValueError("nothing imported")
        
        self.lists[self.name] = mylist

    def make_que(self, value):
        self.lists[self.name].append(value)

    def dequeue(self):
        if self.lists[self.name]:
            rvalue = self.lists[self.name].pop(0)
            self.lists[self.name].append(None)
            return rvalue
        else:
            raise IndexError("empty")
    
    def show(self):
        return self.lists[self.name]
       
#4-2 circle queue
class Cqueue:

    def __init__(self, name, size, *args):
        self.name = name
        self.max_size = size + 3
        self.lists = [None] * self.max_size
        self.front = 0
        self.rear = -1 #-1로 지정하여 마지막 element가 존재하는 곳의 번호를 indexing.
        self.count = 0 

        for arg in args:
            self.enqueue(arg)

    def enqueue(self, value):
        if self.count == self.max_size:
            self.lists[self.name].append(None)
        elif self.count >= self.max_size:
            raise OverflowError("queue is full")
        
        self.rear = (self.rear + 1) % self.max_size
        """etc:: when rear is four, it makes it go to zero"""
        self.lists[self.rear] = value #rear point 증가 후 실행이므로, 그 바로 다음자리에 원소를 input.
        self.count += 1

    def dequeue(self):
        if self.count == 0:
            raise IndexError("empty")
        
        dvalue = self.lists[self.front]
        self.lists[self.front] = None
        self.front = (self.front + 1) % self.max_size
        self.count -= 1
        return dvalue
    
    def show(self):
        if self.count == 0:
            return []
        
        if self.rear >= self.front:
            return self.lists[self.front:self.rear + 1]
        else:
            return self.lists[self.front:] + self.lists[:self.rear + 1]
        #이건 순환한 큐이므로 나누어져서 들어가 삽입되어 있다. 그러므로 앞의 것과 뒤의 것을 합쳐서 원소들을 추출해낸다
        