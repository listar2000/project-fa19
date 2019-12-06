import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from christofides import tsp
import time

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
    flag = start_loc not in homeIndices
    #print("flag is ",flag)
    if flag:
        homeIndices.append(start_loc)
    homeDict = {i:j for i, j in enumerate(homeIndices)}
    #print("home indices are",homeIndices)
    #print("home dict is   :",homeDict)
    homeShortPath = home_shortest_paths(shortPath, homeDict)

    paths, lengths = [], []
    for i in range(10):
        min_length, min_path = float('inf'), None
        for j in range(20):
            length, path = tsp(homeShortPath, len(homeDict))
            if length < min_length:
                min_length = length
                min_path = path
        paths.append(min_path)
        lengths.append(min_length)
    
    path = paths[lengths.index(min(lengths))]

    result, cache, metric_tsp_path = float('inf'), None, None

    for i in range(10):
        length, path = lengths[i], paths[i]
        #print("size of homeIndices is ",homeIndices.__len__()," and the list is ",homeIndices)
        #print(flag,"length of path return by tsp is: ", length, len(path), path,start_loc)
        # translate back to homes' real locations
        #print(start_loc in homeIndices)
        #print("actual path: ",path)
        #print("homeDicts in path:",[homeDict[p] for p in path])
        tmp_path = reorder_visit([homeDict[p] for p in path], start_loc, True)
        #print("afer reordering: ", metric_tsp_path)
        #remove the starting location from the list of homes
        if flag:
            #print("HI")
            tmp_path.pop(0)
            #print(metric_tsp_path)
        # Step 2: P-time dynamic programming algorithm
        tmp_result, tmp_cache = p_time_dp(tmp_path, shortPath, start_loc, adjacency_matrix)
        if tmp_result < result:
            result = tmp_result
            cache = tmp_cache
            metric_tsp_path = tmp_path
    
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

    #print("Final Path is ",finalPath,len(finalPath),finalPath[len(finalPath)-1])
    if finalPath[0] != finalPath[len(finalPath)-1]:
        finalPath.append(finalPath[0])
    #print("Corrected Final Path is ",finalPath)
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
    #print(num_houses,num_of_locations,starting_car_location)
    #print(list_locations)
    #print(list_houses)

    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    #print(len(car_path),car_path)
    #print(len(drop_offs),drop_offs)
    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    # control = int(params[0])

    for input_file in input_files:
        # if control == 2:
        #     if input_file.endswith('_200.in'):
        #         continue
        # else:
        #     if not input_file.endswith('_200.in'):
        #         continue
        #     else:
        #         num = int(input_file[-8])
        #         if num % 2 != control:
        #             continue  
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