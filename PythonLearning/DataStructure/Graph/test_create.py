"""
# @Time    :  2020/8/11
# @Author  :  Jimou Chen
"""


class Vertex:
    def __init__(self, data):
        self.data = data
        self.connect_to = {}  # 记录该点连接的其他边及其权值，如果连接的是v1，那键就是v1

    def add_neighbor(self, side, weight=0):
        self.connect_to[side] = weight

    # 返回该点连接到哪些边和点的信息
    def __str__(self):
        return str(self.data) + ' connect to ' + str(self.connect_to)

    # 返回该点的所有连接点
    def get_connect_ver(self):
        return self.connect_to.keys()

    def get_data(self):
        return self.data

    # 获取该点的某一条连接边的权值
    def get_weight(self, side):
        return self.connect_to[side]


class Graph:
    def __init__(self):
        self.vertex_list = {}
        self.num_vertex = 0

    def add_vertex(self, ver):
        self.num_vertex += 1
        new_vertex = Vertex(ver)
        self.vertex_list[ver] = new_vertex
        return new_vertex

    # 看看点在不在图里面
    def get_vertex(self, ver):
        if ver in self.vertex_list:
            return self.vertex_list[ver]
        else:
            return None

    def add_edge(self, start_ver, end_ver, weight):
        if start_ver not in self.vertex_list:
            self.add_vertex(start_ver)
        if end_ver not in self.vertex_list:
            self.add_vertex(end_ver)

        self.vertex_list[start_ver].add_neighbor(self.vertex_list[end_ver], weight=weight)

    def get_all_ver(self):
        return self.vertex_list.keys()

    # def __contains__(self, ver):
    #     return ver in self.vertex_list


if __name__ == '__main__':
    g = Graph()
    for i in range(6):
        g.add_vertex(i)

    g.add_edge(0, 1, 5)
    g.add_edge(0, 5, 2)
    g.add_edge(1, 2, 4)
    g.add_edge(2, 3, 9)
    g.add_edge(3, 4, 7)
    g.add_edge(3, 5, 3)
    g.add_edge(4, 0, 1)
    g.add_edge(5, 4, 8)
    g.add_edge(5, 2, 1)

