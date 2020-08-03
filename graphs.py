# %%
import algs
from algs import Queue, Stack
import networkx as nx
import matplotlib.pyplot as plt
import copy

# %%
class Graph:
    #classic undirected graph
    def __init__(self, n):
        self.n = n
        self.alist = [[] for i in range(self.n)] #adjacency list
        self.amat = [[0] * (i+1) for i in range(self.n)] #adjacency matrix

    def degree(self, x):
        return len(self.alist[x])

    def connect(self,a,b):
        if b not in self.alist[a]:
            self.alist[a].append(b)
            self.alist[b].append(a)

        self.amat[max(a,b)][min(a,b)] = 1

    def disconnect(self, a, b):
        if b in self.alist[a]:
            self.alist[a].remove(b)
            self.alist[b].remove(a)

        self.amat[max(a,b)][min(a,b)] = 0

    def bfs(self,s):
        # BFS using colors white, grey, and black
        #   White: undiscovered
        #   Grey: discovered, may have undiscovered neighbors
        #   Black: discovered, all neighbors are discovered
        color = distance = [-1] * self.n # -1: white, 0: grey, 1: black
        distance = [None] * self.n #distance from source s
        prev = [None] * self.n #predecessors of nodes

        color[s] = 0
        distance[s] = 0
        prev[s] = None #redundant

        Q = Queue()
        Q.enqueue(s)
        while not Q.isEmpty():
            u = Q.dequeue()
            for v in self.alist[u]:
                if color[v] == -1:
                    color[v] = 0
                    distance[v] = distance[u] + 1
                    prev[v] = u
                    Q.enqueue(v)
            color[u] = 1
        return (color, distance, prev) #returns results of search

    def dfs(self,s):
        # DFS using colors white, grey, and black
        #   White: undiscovered
        #   Grey: discovered, may have undiscovered neighbors
        #   Black: discovered, all neighbors are discovered
        color = distance = [-1] * self.n # -1: white, 0: grey, 1: black
        distance = [None] * self.n #distance from source s
        prev = [None] * self.n #predecessors of nodes

        color[s] = 0
        distance[s] = 0
        prev[s] = None #redundant

        S = Stack()
        S.push(s)
        while not S.isEmpty():
            u = S.pop()
            for v in self.alist[u]:
                if color[v] == -1:
                    color[v] = 0
                    distance[v] = distance[u] + 1
                    prev[v] = u
                    S.push(v)
            color[u] = 1
        return (color, distance, prev) #returns results of search

    def print_path(self,s,v):
        search = self.bfs(s)
        if v == s:
            print(s)
        elif search[2][v] == None:
            print("No path exists from " + str(s) + " to " + str(v))
        else:
            self.print_path(s, search[2][v])
            print(v)

    def show(self):
        g = nx.Graph()
        [g.add_node(n) for n in range(self.n)]
        for v in range(self.n):
            for u in self.alist[v]:
                g.add_edge(u,v)
        nx.draw(g,with_labels=True,node_color=['orange'])
        plt.show()

    def is_connected(self):
        search = self.bfs(0)
        return search[0].count(1) == self.n

    def is_tree(self):
        return self.is_connected() and sum([x.count(1) for x in self.amat]) == self.n - 1

    @staticmethod
    def test():
        a = Graph(5)
        a.connect(4,1)
        a.connect(2,4)
        a.connect(3,0)

        a.print_path(2,3)
        a.show()

class DirectedGraph(Graph):
    #classic undirected graph
    def __init__(self,n):
        super(DirectedGraph,self).__init__(n)
        self.amat = [[0]*self.n for i in range(self.n)] #full adjacency matrix

    def connect(self,a,b):
        self.alist[a].append(b)
        self.amat[a][b] = 1

    def disconnect(self,a,b):
        if b in self.alist[a]:
            self.alist[a].remove(b)
        self.amat[a][b] = 0

    def show(self):
        g = nx.DiGraph()
        [g.add_node(n) for n in range(self.n)]
        for v in range(self.n):
            for u in self.alist[v]:
                g.add_edge(u,v)
        nx.draw(g,with_labels=True,node_color=['orange'])
        plt.show()

    @staticmethod
    def test():
        b = DirectedGraph(5)
        b.connect(2,3)
        b.connect(0,3)
        b.connect(3,4)

        b.print_path(2,4)
        b.show()

# %%
def prufer_decode(seq):
    n = len(seq) + 2
    G = Graph(n)
    P = list(seq)
    V = list(range(0,n))
    for i in range(n-2):
        v = min([vertex for vertex in V if vertex not in P[i:] ])
        G.connect(v,P[i])
        V.remove(v)
    G.connect(V[0],V[1])
    return G

def prufer_encode(G):
    G = copy.deepcopy(G)
    n = G.n
    seq = []
    for i in range(n-2):
        v = min([vertex for vertex in range(n) if G.degree(vertex)==1])
        neighbor = G.alist[v][0]
        seq.append(neighbor)
        G.disconnect(v,neighbor)
    return tuple(seq)

def test_prufer():
    c = prufer_decode((4,0,0,4))

    c.show()

    print(prufer_encode(c))

#test_prufer()
