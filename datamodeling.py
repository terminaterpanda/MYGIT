
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
        #이건 순환한 큐이므로 나누어져서 들어가 삽입되어 있다. 그러므로 앞의 것과 뒤의 것을 합쳐서 원소들을 추출해낸다.

#5.Hash table

class Hashtable:

    def __init__(self, name, *args):
        self.name = name
        #n = range(len(args)) <enumerate> 함수의 존재로 n = 이런 식으로 할 필요가 없다.
        self.lists = {}

        for i, value in enumerate(args):
            hash_key = self.hashfunc(value, len(args))
            self.lists[hash_key] = value

    def hashfunc(self, value, size):
        return hash(value) % size
    
    def show_table(self):

        for key, value in self.lists.items():
            print(f"key: {key} -> value: {value}")
#하지만 이 function은 hash함수에 크게 의존한다는 단점이 존재.

#5. Hash table(if hash not)::

class MyHashTable:
    
    def __init__(self, name, *args):
        self.name = name
        self.lists = {}
        if not args:
            raise ValueError("nothing is inserted")

        for i, value in enumerate(args, start=1):
            self.lists[i] = value #self.lists[i] 이렇게 순서대로 value값으로 저장.
            #dictionary == {key-value}값 형식으로 저장되어 있으므로 딕셔너리에서 data에 접근하기 위한 식별자 자체가 key

    def get(self, key):
        return self.lists.get(key, "key not found")
    #get 메서드 == 첫번쨰 인자가 찾을 키. 두번째 인자는 못찾을 경우 반환할 기본값.
      
    def show_table(self):
        for key, value in self.lists.items(): #키와 값을 쌍으로 반환.(모든 항목에 대해 iterate)
            print(f"key: {key} -> value: {value}")


#6. Graph

"""directed node"""

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None #이건 다음 노드로의 연결을 정의하고, 기본값을 none.

class MakeNode:
    def __init__(self, *args):
        self.lists = {}

        if not args:
            raise ValueError("nothing is imported")
        
        for i, value in enumerate(args, start=1):
            node = Node(value)
            self.lists[i] = node 

    def connect(self):
        nodekey = list(self.lists.keys())
        #self.lists는 딕셔너리이므로 key값을 가지고 있고, 그 key갑들만 가져와서 list로 생성.

        for i in range(1, len(nodekey)):
            self.lists[i].next = self.lists[i+1]
        #반복문으로 node들을 전부 한방향으로 연결.

class Graph:
    def __init__(self, make_node):
        self.start_node = make_node.lists[1]

    def display(self):
        current = self.start_node
        #while 반복문을 use해서 next로 쭉 가서 마지막까지 간다.
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("nothing") #더이상 반복할 것이 없는 경우에서 이 list형식을 표시.


#6-1 Graph -1

class Node1:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.before = None

class MakeNode2:
    def __init__(self, *args):
        self.lists = {}

        if not args:
            raise ValueError("nothing is implemented")
        
        for i, value in enumerate(args, start=1):
            node = Node1(value)
            self.lists[i] = node
        print("created nodes:", self.lists)

    def connect_nodes(self, from_idx, to_idx):
        """node index -> connect them"""
        if from_idx in self.lists and to_idx in self.lists:
            self.lists[from_idx].next = self.lists[to_idx]
            self.lists[to_idx].before = self.lists[from_idx]
            print("node1 {from_idx} -> node2 {to_idx}")
        else:
            raise IndexError("error occured")


#7. Tree

import random
class Node:
    def __init__(self, data):
        if not data:
            raise ValueError("error")
        self.data = data
        self.neighbors = []

class MakeTreeNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("Error")
        self.lists = []
        self.root = None

        for value in args:
            tree_node = Node(value)
            self.lists.append(tree_node)
            #이런 식으로 한 경우에 Tree로 value 인자들을 받아서 구현한다.
            
    def ChooseRoot(self):
        n = len(self.lists)
        if n == 0:
            raise ValueError("Error")

        root_index = random.randint(0, n-1)
        self.root = self.lists[root_index]

        return self.root
    
    def CreateLayers(self, *args):
        if self.root is None:
            raise ValueError("Error: no root node")
        
        layers = [[self.root]] #이중 list 구조
        remain_nodes = [node for node in self.lists if node != self.root]
        #remain_nodes에는 self.root 제외되어 있음.
        
        if sum(args) != len(remain_nodes):
            raise ValueError("Error : sum is not properly defined")
        
        layers = []
        start_index = 0

        for count in args: #count == 현재 층에 포함딜 node 개수
            if start_index < len(remain_nodes):
                end_index = start_index + count
                if end_index > len(remain_nodes):
                    end_index = len(remain_nodes)

                layers_nodes = remain_nodes[start_index:end_index]
                #start_index부터 end_index를 해서 이런 식으로 연결을 해준다.
                layers.append(layers_nodes)

                start_index = end_index
            
        return layers
    #hasattr == "변수(object)의 존재 여부를 확인하는 함수이다.(bool형태로 이루어진다.)"
    def ConnectLayers(self, layers):
        for i in range(len(layers)-1):
            for parent in layers[i]:#layers[i]의 각 노드가 부모 node로 취급된다.
                for child in layers[i + 1]:
                    parent.neighbors.append(child)


tree = MakeTreeNode(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # 10개의 노드 생성
tree.ChooseRoot()
layers = tree.CreateLayers(2, 3, 4)  # 층을 2-3-4로 생성
tree.ConnectLayers(layers)  # 층 연결

for i, layer in enumerate(layers):
    print(f"Layer {i + 1}:")
    for node in layer:
        neighbors = [neighbor.data for neighbor in node.neighbors]
        print(f"Node {node.data} -> Neighbors: {neighbors}")


# 8. Heap

class Node3:
    def __init__(self, data):
        if not data:
            raise ValueError("error occured")
        self.data = data
        self.border = []


class HeapNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("args are not imported")
        self.lists = [] #self.lists 라는 list를 정의
        self.root = None #self.root 값을 만들고 none을 기본값으로 두기

        for value in args:
            heap_node = Node3(value) #args의 value개수만큼 heap_node -> self.lists 안에 추가
            self.lists.append(heap_node)

    def MakeHeapNode(self):
        n = len(self.lists)
        if n == 0:
            raise ValueError("error occured")
        self.root = max(self.lists, key = lambda x: x.data)
        #lambda == 파이썬 익명 함수 define.(x == Node3)
        return self.root

    def CreateNode(self, *args):
        if self.root is None:
            raise ValueError("error occured")
        
        layers = [[self.root]] #이중 list를 생성해놓음.
        remain = [node for node in self.lists if node != self.root]

        if sum(args) != len(remain):
            raise ValueError("error")
        """lambda 함수는 값을 바로 반환하고 lambda(인자, 함수) == a로 할당하여 불러온다.
        (한줄로 간단하게 정의할 때 use.)"""
        layers = []
        start = 0

        for count in args:
            if start < len(remain):
                end_index = start + count
                if end_index > len(remain):
                    end_index = len(remain)

                layers_nodes = sorted(remain[start:end_index], key=lambda x: x.data, reverse=True)

                layers.append(layers_nodes)
                start = end_index

        return layers #층 쌓기는 완료.
    
    def Connect(self, layers):
        for i in range(self,layers):
            for parent in layers[i]:
                for child in layers[i + 1]:
                    parent.neighbors.append(child)

    def SelfSort(self):
        queue = [self.root]
        sorted_values = []

        while queue:
            now = queue.pop(0)
            sorted_values.append(now.data)

            for neighbor in now.neighbors:
                queue.append(neighbor)

        return sorted_values
"""
Sorted() == key(sorted) 함수에서 use되는 속성값.
lambda(X) == 익명함수 -> 즉, 일회성으로 넣고 함수를 저장하지 않음으로 불필요한 연산을 최소화시키는것이 목적.
"""

