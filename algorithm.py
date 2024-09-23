# 1. BFS

# 1-1. BFS [with no collections package]

from random import shuffle

class Node:
    def __init__(self, data):
        if not data:
            raise ValueError("nothing is imported")
        self.data = data
        self.neighbors = [] #다중연결 == "list"
        
class MakeNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("nothing is imported")
        self.lists = {}
        
        for i, value in enumerate(args, start=1):
            current = Node(value) #전달 value값을 이용해서 node 객체 생성
            self.lists[i] = current #node를 dictionary self.lists에 저장

    def connect(self, key1, key2):
        node1 = self.lists[key1]
        node2 = self.lists[key2]

        node1.neighbors.append(node2)
        node2.neighbors.append(node1) #양방향 연결

    def bfs(self, start_key):
        if start_key not in self.lists:
            raise ValueError("error")
        
        start_node = self.lists[start_key]
        visited = set() #visted를 set(집합) 형식으로 만들어서 value 값 부여.
        queue = [start_node]

        while queue: #queue가 비어있지 않은 한 끊임없이 repeat.
            current_node = queue.pop(0)
            
            if current_node not in visited:
                print(f"node: {current_node.data}")
                visited.add(current_node)
                #node를 visited <set(집합)> 에 추가하여 중복 방지.

                for neighbor in current_node.neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

        if not queue:
            print("visited everythng")
            queue.clear()
           

# 1-2. BFS [collections package]

from collections import deque

def bfs (graph, node, visited):
    queue = deque([node])
    visited[node] = True

    while queue:
        v = queue.popleft()
        print(v, end=" ")
        #처리 중 node에서 방문하지 않은 node(인접) 전부 queue에 삽입
        for i in graph[v]:
            if not (visited[i]):
                queue.append(i)
                visited[i] = True

graph = [
    [],       
    [2, 3],   
    [1, 8],   
    [1, 5],   
    [],       
    [3],      
    [],      
    [],       
    [2]       
]

visited = [False] * 9

bfs(graph, 1, visited)
