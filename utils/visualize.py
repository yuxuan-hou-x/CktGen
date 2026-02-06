import os
import gzip
import igraph
import pygraphviz as pgv
import sys
# from PIL import Image


SUBG_NODE = {
    0: ['In'],
    1: ['Out'],
    2: ['R'],
    3: ['C'],
    4: ['R', 'C'],
    5: ['R', 'C'],
    6: ['+gm+'],
    7: ['-gm+'],
    8: ['+gm-'],
    9: ['-gm-'],
    10: ['C', '+gm+'],
    11: ['C', '-gm+'],
    12: ['C', '+gm-'],
    13: ['C', '-gm-'],
    14: ['R', '+gm+'],
    15: ['R', '-gm+'],
    16: ['R', '+gm-'],
    17: ['R', '-gm-'],
    18: ['C', 'R', '+gm+'],
    19: ['C', 'R', '-gm+'],
    20: ['C', 'R', '+gm-'],
    21: ['C', 'R', '-gm-'],
    22: ['C', 'R', '+gm+'],
    23: ['C', 'R', '-gm+'],
    24: ['C', 'R', '+gm-'],
    25: ['C', 'R', '-gm-'],
}


# subgraph connection way for each node
SUBG_CON = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: 'series',
    5: 'parral',
    6: None,
    7: None,
    8: None,
    9: None,
    10: 'parral',
    11: 'parral',
    12: 'parral',
    13: 'parral',
    14: 'parral',
    15: 'parral',
    16: 'parral',
    17: 'parral',
    18: 'parral',
    19: 'parral',
    20: 'parral',
    21: 'parral',
    22: 'series',
    23: 'series',
    24: 'series',
    25: 'series',
}


SUBG_INDI = {
    0: [],
    1: [],
    2: [0],
    3: [1],
    4: [0, 1],
    5: [0, 1],
    6: [2],
    7: [2],
    8: [2],
    9: [2],
    10: [1, 2],
    11: [1, 2],
    12: [1, 2],
    13: [1, 2],
    14: [0, 2],
    15: [0, 2],
    16: [0, 2],
    17: [0, 2],
    18: [1, 0, 2],
    19: [1, 0, 2],
    20: [1, 0, 2],
    21: [1, 0, 2],
    22: [1, 0, 2],
    23: [1, 0, 2],
    24: [1, 0, 2],
    25: [1, 0, 2],
}


def draw_network(g, path, backbone=False, start_symbol=False):
    graph = pgv.AGraph(
        directed=True, strict=True, fontname='Helvetica', arrowtype='open'
    )

    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return

    for idx in range(g.vcount()):
        if start_symbol:
            add_node_start_symbol(graph, idx, g.vs[idx]['type'], g.vs[idx]['path'])
        else:
            add_node(graph, idx, g.vs[idx]['type'], g.vs[idx]['path'])

    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx - 1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)

    graph.layout(prog='dot')
    graph.draw(path)


def add_node_start_symbol(graph, node_id, node_type, node_path, shape='box', style='filled'):
    if node_type == 0:
        node_type = '0: input'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 1:
        node_type = '1: output'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 2:
        node_type = 'Start Symbol'
        if node_path in [2, 3, 4]:
            color = 'green'
        else:
            color = 'skyblue'
    elif node_type == 3:
        node_type = '2: R'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 4:
        node_type = '3: C'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 5:
        node_type = '4: [R, C], series'
        # 4: ['R', 'C'],
        # 4: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 6:
        node_type = '5: [R, C], parral'
        # 5: ['R', 'C'],
        # 5: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 7:
        node_type = '6: +gm+'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 8:
        node_type = '7: -gm+'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 9:
        node_type = '8: +gm-'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 10:
        node_type = '9: -gm-'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 11:
        node_type = '10: [C, +gm+], parral'
        # 10: ['C', '+gm+'],
        # 10: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 12:
        node_type = '11: [C, -gm+], parral'
        # 11: ['C', '-gm+'],
        # 11: 'parral',
        
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 13:
        node_type = '12: [C, +gm-], parral'
        # 12: ['C', '+gm-'],
        # 12: 'parral',
        
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 14:
        node_type = '13: [C, -gm-], parral'
        # 13: ['C', '-gm-'],
        # 13: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 15:
        node_type = '14: [R, +gm+], parral'
        # 14: ['R', '+gm+'],
        # 14: 'parral',
        
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 16:
        node_type = '15: [R, -gm+], parral'
        # 15: ['R', '-gm+'],
        # 15: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 17:
        node_type = '16: [R, +gm-], parral'
        # 16: ['R', '+gm-'],
        # 16: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 18:
        node_type = '17: [R, -gm-], parral'
        # 17: ['R', '-gm-'],
        # 17: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 19:
        node_type = '18: [C, R, +gm+], parral'
        # 18: ['C', 'R', '+gm+'],
        # 18: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 20:
        node_type = '19: [C, R, -gm+], parral'
        # 19: ['C', 'R', '-gm+'],
        # 19: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 21:
        node_type = '20: [C, R, +gm-], parral'
        # 20: ['C', 'R', '+gm-'],
        # 20: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 22:
        node_type = '21: [C, R, -gm-], parral'
        # 21: ['C', 'R', '-gm-'],
        # 21: '',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 23:
        node_type = '22: [C, R, +gm+], series'
        # 22: ['C', 'R', '+gm+'],
        # 22: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 24:
        node_type = '23: [C, R, -gm+], series'
        # 23: ['C', 'R', '-gm+'],
        # 23: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 25:
        node_type = '24: [C, R, +gm-], series'
        # 24: ['C', 'R', '+gm-'],
        # 24: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 26:
        node_type = '25: [C, R, -gm-], series'
        # 25: ['C', 'R', '-gm-'],
        # 25: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'

    # node_type = f"{node_type}\n({node_id})"
    node_prop = f"({node_id})\n{node_type}"
    graph.add_node(
        node_id,
        label=node_prop,
        color='black',
        fillcolor=color,
        shape=shape,
        style=style,
        fontsize=24,
    )


def add_node(graph, node_id, node_type, node_path, shape='box', style='filled'):
    if node_type == 0:
        node_type = '0: input'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 1:
        node_type = '1: output'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 2:
        node_type = '2: R'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 3:
        node_type = '3: C'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 4:
        node_type = '4: [R, C], series'
        # 4: ['R', 'C'],
        # 4: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 5:
        node_type = '5: [R, C], parral'
        # 5: ['R', 'C'],
        # 5: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 6:
        node_type = '6: +gm+'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 7:
        node_type = '7: -gm+'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 8:
        node_type = '8: +gm-'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 9:
        node_type = '9: -gm-'
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 10:
        node_type = '10: [C, +gm+], parral'
        # 10: ['C', '+gm+'],
        # 10: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 11:
        node_type = '11: [C, -gm+], parral'
        # 11: ['C', '-gm+'],
        # 11: 'parral',
        
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 12:
        node_type = '12: [C, +gm-], parral'
        # 12: ['C', '+gm-'],
        # 12: 'parral',
        
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 13:
        node_type = '13: [C, -gm-], parral'
        # 13: ['C', '-gm-'],
        # 13: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 14:
        node_type = '14: [R, +gm+], parral'
        # 14: ['R', '+gm+'],
        # 14: 'parral',
        
        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 15:
        node_type = '15: [R, -gm+], parral'
        # 15: ['R', '-gm+'],
        # 15: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 16:
        node_type = '16: [R, +gm-], parral'
        # 16: ['R', '+gm-'],
        # 16: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 17:
        node_type = '17: [R, -gm-], parral'
        # 17: ['R', '-gm-'],
        # 17: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 18:
        node_type = '18: [C, R, +gm+], parral'
        # 18: ['C', 'R', '+gm+'],
        # 18: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 19:
        node_type = '19: [C, R, -gm+], parral'
        # 19: ['C', 'R', '-gm+'],
        # 19: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 20:
        node_type = '20: [C, R, +gm-], parral'
        # 20: ['C', 'R', '+gm-'],
        # 20: 'parral',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 21:
        node_type = '21: [C, R, -gm-], parral'
        # 21: ['C', 'R', '-gm-'],
        # 21: '',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 22:
        node_type = '22: [C, R, +gm+], series'
        # 22: ['C', 'R', '+gm+'],
        # 22: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 23:
        node_type = '23: [C, R, -gm+], series'
        # 23: ['C', 'R', '-gm+'],
        # 23: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 24:
        node_type = '24: [C, R, +gm-], series'
        # 24: ['C', 'R', '+gm-'],
        # 24: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'
    elif node_type == 25:
        node_type = '25: [C, R, -gm-], series'
        # 25: ['C', 'R', '-gm-'],
        # 25: 'series',

        if node_path in [2, 3, 4]:
            color = 'pink'
        else:
            color = 'skyblue'

    # node_type = f"{node_type}\n({node_id})"
    node_prop = f"({node_id})\n{node_type}"
    graph.add_node(
        node_id,
        label=node_prop,
        color='black',
        fillcolor=color,
        shape=shape,
        style=style,
        fontsize=24,
    )


def plot_graph(args, g, name, backbone=False, pdf=False, pace=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(args['out_dir'], name + '.png')

    if pdf:
        file_name = os.path.join(args['out_dir'], name + '.pdf')

    # print(g, file_name, backbone)
    if pace:
        draw_network(g, file_name, backbone, start_symbol=True)
    else:
        draw_network(g, file_name, backbone, start_symbol=False)

    return file_name