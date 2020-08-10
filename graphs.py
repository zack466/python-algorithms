# %%
# adds graphing support when matplotlib is not available (e.g. using PyPy)
try:
    import matplotlib.pyplot as plt
    using_matplotlib = False
except:
    using_matplotlib = False

# backupt graphing support
import pydot
from IPython.display import Image, display
def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

# imports
import algs
from algs import Queue, Stack, BinaryHeap
import networkx as nx
import copy
from collections import deque

# %%
class Graph:
    #classic undirected graph
    def __init__(self, n):
        self.n = n
        self.alist = [[] for i in range(self.n)] #adjacency list
        self.amat = [[0] * (i+1) for i in range(self.n)] #adjacency matrix (only diagonal and below to save space)

    def degree(self, x):
        # returns the degree of a node x
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
        color = [-1] * self.n # -1: white, 0: grey, 1: black
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

    def dfs_rec(self, s, visited=None):
        #recursive dfs
        if visited==None:
            visited = [False] * self.n
        visited[s] = True
        for node in self.alist[s]:
            if not visited[node]:
                self.dfs_rec(node, visited)
        return visited

    def print_path(self,s,v):
        # prints the path from node s to node v or states that one does not exist
        search = self.bfs(s)
        if v == s:
            print(s)
        elif search[2][v] == None:
            print("No path exists from " + str(s) + " to " + str(v))
        else:
            self.print_path(s, search[2][v])
            print(v)

    def show(self):
        # visualizes graph using either matplotlib or pydot
        g = nx.Graph()
        [g.add_node(n) for n in range(self.n)]
        for v in range(self.n):
            for u in self.alist[v]:
                g.add_edge(u,v)
        if using_matplotlib:
            nx.draw(g,with_labels=True,node_color=['orange'])
            plt.show()
        else:
            pdot = nx.drawing.nx_pydot.to_pydot(g)
            view_pydot(pdot)

    def connected_components(self):
        # returns a list of all connected components in the graph
        visited = [False] * self.n
        counter = 0
        components = []
        for vertex in range(self.n):
            if not visited[vertex]:
                components.append([])
                connected_to = self.dfs_rec(vertex)
                for i in range(len(connected_to)):
                    if connected_to[i]:
                        visited[i] = True
                        components[-1].append(i)
        return components

    def _is_cyclic(self, s=0, visited=None, prev=None):
        # returns whether the connect component containing s is cyclic or not
        if visited == None:
            visited = [False] * self.n
        if visited[s]:
            return True
        else:
            visited[s] = True
        for node in self.alist[s]:
            if node != prev:
                return self._is_cyclic(node, visited, s)
        return False

    def is_cyclic(self):
        # returns whether a cycle exists in this graph or not
        for v in range(self.n):
            if self._is_cyclic(v):
                return True
        return False

    def is_connected(self):
        search = self.bfs(0)
        return search[0].count(1) == self.n

    def is_tree(self):
        return self.is_connected() and sum([x.count(1) for x in self.amat]) == self.n - 1

    def _is_bipartite(self, s):
        # returns whether connected component containing s is bipartite or not
        color = [None] * self.n # two colors: 0 and 1
        color[s] = 0
        Q = Queue()
        Q.enqueue(s)
        bipartite = True
        while not Q.isEmpty() and bipartite:
            u = Q.dequeue()
            for v in self.alist[u]:
                if color[v] == None:
                    color[v] = 1 - color[u]
                    Q.enqueue(v)
                elif color[v] == color[u]:
                    bipartite = False
        return bipartite

    def is_bipartite(self):
        for v in range(self.n):
            if not self._is_bipartite(v):
                return False
        return True

    @staticmethod
    def test():
        a = Graph(9)
        a.connect(0,1)
        a.connect(2,3)
        a.connect(1,3)
        a.connect(3,4)
        a.connect(6,7)
        a.connect(6,8)
        print(a.connected_components())
        print(a.is_bipartite())
        a.show()

# Graph.test()

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
        # visualizes graph using either matplotlib or pydot
        g = nx.DiGraph()
        [g.add_node(n) for n in range(self.n)]
        for v in range(self.n):
            for u in self.alist[v]:
                g.add_edge(v,u)
        if using_matplotlib:
            nx.draw(g,with_labels=True,node_color=['orange'])
            plt.show()
        else:
            pdot = nx.drawing.nx_pydot.to_pydot(g)
            view_pydot(pdot)

    def _topological_sort(self, s, visited, ts):
        visited[s] = True
        for v in self.alist[s]:
            if not visited[v]:
                self._topological_sort(v, visited, ts)
        ts.appendleft(s)

    def topological_sort(self):
        # returns a topologically sorted list of nodes
        assert not self.is_cyclic()
        ts = deque() # use deque for efficient left-append
        visited = [False] * self.n
        for i in range(self.n):
            if not visited[i]:
                self._topological_sort(i, visited, ts)
        return list(ts)

    def kahn_topsort(self):
        #alternate algorithm for topological sort
        amat = copy.deepcopy(self.amat)
        assert not self.is_cyclic()
        pq = BinaryHeap()
        ts = []
        for v in range(self.n):
            if [amat[x][v] for x in range(self.n)].count(1) == 0: #if indegree is 0
                pq.insert(v)
        while not pq.isEmpty():
            v = pq.deleteMax()
            ts.append(v)
            for i in range(self.n):
                amat[v][i] = 0
            for v in range(self.n):
                if v not in ts and v not in pq:
                    if [amat[x][v] for x in range(self.n)].count(1) == 0:
                        pq.insert(v)
        return ts

    @staticmethod
    def test():
        a = DirectedGraph(8)
        a.connect(0,1)
        a.connect(0,2)
        a.connect(1,2)
        a.connect(1,3)
        a.connect(2,3)
        a.connect(3,4)
        a.connect(2,5)
        a.connect(7,6)
        print(a.topological_sort())
        a.show()

# DirectedGraph.test()

# %%
def prufer_decode(seq):
    # returns the graph represented by a prufer sequence
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
    # encodes a tree as a prufer sequence
    assert G.is_tree(), "graph G must be a tree"
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

# test_prufer()
