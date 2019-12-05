import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from christofides import tsp

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    start_loc = list_of_locations.index(starting_car_location)

    # Step 1: Construct a shortest dist matrix from adjacancy matrix
    shortPath = get_shortest_dist_matrix(adjacency_matrix)
    homeIndices = [list_of_locations.index(home) for home in list_of_homes]

    if start_loc not in homeIndices:
        homeIndices.append(start_loc)
    homeDict = {i:j for i, j in enumerate(homeIndices)}

    homeShortPath = home_shortest_paths(shortPath, homeDict)
    length, path = tsp(homeShortPath, len(homeDict))

    # translate back to homes' real locations
    metric_tsp_path = reorder_visit([homeDict[p] for p in path], start_loc, start_loc in homeIndices)

    # Step 2: P-time dynamic programming algorithm
    result, cache = p_time_dp(metric_tsp_path, shortPath, start_loc, adjacency_matrix)

    k = len(metric_tsp_path)
    finalPath = [start_loc]
    dropOffDict = {}
    loc = start_loc
    while k > 0:
        loc = cache[(k, loc)]
        k -= 1
        finalPath.append(loc)
        if loc not in dropOffDict:
            dropOffDict[loc] = []
        dropOffDict[loc].append(metric_tsp_path[k])

    # Step3: remove duplicates and recover the entire path
    finalPath = recover_entire_path(finalPath, adjacency_matrix)
    return finalPath, dropOffDict
"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='outputs', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
