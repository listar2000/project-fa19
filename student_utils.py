import networkx as nx
import numpy as np


def decimal_digits_check(number):
    number = str(number)
    parts = number.split('.')
    if len(parts) == 1:
        return True
    else:
        return len(parts[1]) <= 5


def data_parser(input_data):
    number_of_locations = int(input_data[0][0])
    number_of_houses = int(input_data[1][0])
    list_of_locations = input_data[2]
    list_of_houses = input_data[3]
    starting_location = input_data[4][0]

    adjacency_matrix = [[entry if entry == 'x' else float(entry) for entry in row] for row in input_data[5:]]
    return number_of_locations, number_of_houses, list_of_locations, list_of_houses, starting_location, adjacency_matrix


def adjacency_matrix_to_graph(adjacency_matrix):
    node_weights = [adjacency_matrix[i][i] for i in range(len(adjacency_matrix))]
    adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in adjacency_matrix]

    for i in range(len(adjacency_matrix_formatted)):
        adjacency_matrix_formatted[i][i] = 0

    G = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix_formatted))

    message = ''

    for node, datadict in G.nodes.items():
        if node_weights[node] != 'x':
            message += 'The location {} has a road to itself. This is not allowed.\n'.format(node)
        datadict['weight'] = node_weights[node]

    return G, message


def is_metric(G):
    shortest = dict(nx.floyd_warshall(G))
    for u, v, datadict in G.edges(data=True):
        if abs(shortest[u][v] - datadict['weight']) >= 0.00001:
            return False
    return True


def adjacency_matrix_to_edge_list(adjacency_matrix):
    edge_list = []
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[0])):
            if adjacency_matrix[i][j] == 1:
                edge_list.append((i, j))
    return edge_list


def is_valid_walk(G, closed_walk):
    if len(closed_walk) == 2:
        return closed_walk[0] == closed_walk[1]
    return all([(closed_walk[i], closed_walk[i+1]) in G.edges for i in range(len(closed_walk) - 1)])


def get_edges_from_path(path):
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]

"""
G is the adjacency matrix.
car_cycle is the cycle of the car in terms of indices.
dropoff_mapping is a dictionary of dropoff location to list of TAs that got off at said droppoff location
in terms of indices.
"""
def cost_of_solution(G, car_cycle, dropoff_mapping):
    cost = 0
    message = ''
    dropoffs = dropoff_mapping.keys()
    if not is_valid_walk(G, car_cycle):
        message += 'This is not a valid walk for the given graph.\n'
        cost = 'infinite'

    if not car_cycle[0] == car_cycle[-1]:
        message += 'The start and end vertices are not the same.\n'
        cost = 'infinite'
    if cost != 'infinite':
        if len(car_cycle) == 1:
            car_cycle = []
        else:
            car_cycle = get_edges_from_path(car_cycle[:-1]) + [(car_cycle[-2], car_cycle[-1])]
        if len(car_cycle) != 1:
            driving_cost = sum([G.edges[e]['weight'] for e in car_cycle]) * 2 / 3
        else:
            driving_cost = 0
        walking_cost = 0
        shortest = dict(nx.floyd_warshall(G))

        for drop_location in dropoffs:
            for house in dropoff_mapping[drop_location]:
                walking_cost += shortest[drop_location][house]

        message += f'The driving cost of your solution is {driving_cost}.\n'
        message += f'The walking cost of your solution is {walking_cost}.\n'
        cost = driving_cost + walking_cost

    message += f'The total cost of your solution is {cost}.\n'
    return cost, message


def convert_locations_to_indices(list_to_convert, list_of_locations):
    return [list_of_locations.index(name) if name in list_of_locations else None for name in list_to_convert]

"""
Below are the code replicating previous code for pairwise shortest dist algorithm
"""
def get_shortest_dist_matrix(adjacency_matrix):

    neighbors = get_neighbor_from_matrix(adjacency_matrix)

    length = len(adjacency_matrix)
    shortPath = dict()

    for i in range(length):
        shortPath[(i, i)] = 0

    indirect = dict()
    for i in range(length - 1):
        indirect[i] = []
        for j in range(i + 1, length):
            if j not in neighbors[i]:
                indirect[i].append(j)
                shortPath[(i, j)] = float('inf')
            else:
                shortPath[(i, j)] = adjacency_matrix[i][j]
    
    for k in range(length):
        for i in range(length - 1):
            for j in indirect[i]:
                if k != i and k != j:
                    tmpDist = shortestDist(shortPath, i, k) + shortestDist(shortPath, k, j)
                    shortPath[(i, j)] = min(shortPath[(i, j)], tmpDist)

    return shortPath

def shortestDist(shortPath, u, v):
    if u is v:
        return 0
    elif u < v:
        return shortPath[(u, v)]
    else:
        return shortPath[(v, u)]

def get_neighbor_from_matrix(matrix):
    neighbors = dict()
    for index, row in enumerate(matrix):
        neighbors[index] = [i for i, j in enumerate(row) if j is not 'x']
    return neighbors

def home_shortest_paths(shortPath, homeDict):
    homePath = {}

    length = len(homeDict)

    for u in range(length):
        for v in range(u, length):
            i = homeDict[u]
            j = homeDict[v]
            if i < j:
                homePath[(u, v)] = shortPath[(i, j)]
            else:
                homePath[(u, v)] = shortPath[(j, i)]
    
    return homePath

"""
Below are the codes for the polynomial-time dynamic programming algorithm for 4/3 approximation
"""
def p_time_dp(visit_order_list, shortPath, start_loc, matrix):
    homeLength = len(visit_order_list)
    length = len(matrix)
    #print("number of houses is ",homeLength)
    energy = {}
    i = 0
    energy[i] = {}
    for j in range(length):
        energy[i][j] = 2/3 *shortestDist(shortPath, start_loc, j)
    
    cache = {}

    for i in range(1, homeLength + 1):
        energy[i] = {}
        for j in range(length):
            energy[i][j] = float('inf')
            for k in range(length):
                tmp = energy[i - 1][k] + shortestDist(shortPath, k, visit_order_list[i - 1]) + 2/3 * shortestDist(shortPath, j, k)
                if tmp < energy[i][j]:
                    energy[i][j] = tmp
                    cache[(i, j)] = k  

    return energy[homeLength][start_loc], cache

"""
The below codes are for the final recovery of the entire path
"""
def recover_entire_path(dropOffLocs, adjacency_matrix):
    # remove duplicates in dropOffLocs
    uniqueDropOffLocs = [dropOffLocs[0]]
    for i in range(1, len(dropOffLocs)):
        if not dropOffLocs[i] == dropOffLocs[i - 1]:
            uniqueDropOffLocs.append(dropOffLocs[i])

    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    i, j = 0, 1
    recovered_path = []
    while j < len(uniqueDropOffLocs):
        dijkstra_path = nx.algorithms.dijkstra_path(G, uniqueDropOffLocs[i], uniqueDropOffLocs[j])
        recovered_path += dijkstra_path[:-1]
        i += 1
        j += 1
    # add the starting location at the end
    recovered_path.append(uniqueDropOffLocs[-1])
    return recovered_path

"""
"""

def reorder_visit(metric_tsp_path: list, start_loc: int, contains_start: bool):
    reordered_path = []
    if contains_start:
        reordered_path.append(start_loc)
    headIndex = metric_tsp_path.index(start_loc) + 1
    reordered_path.extend(metric_tsp_path[headIndex:])
    reordered_path.extend(metric_tsp_path[:headIndex - 1])
    return reordered_path