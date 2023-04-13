#!/usr/bin/python3
import math

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import sys


def branch_and_bound_initialize_array(cities: list[City]):
    """
    Initializes the first array of values with the original distances between the cities.

    Time complexity:
    Iterates n times for each city, indicating the distance from that city to the other.
    Time: O(n^2)

    Space Complexity:
    The table is n x n with a distance in each value representing the edge between those two cities.
    Space: O(n^2)

    """
    array = []
    length = len(cities)
    for i in range(0, length):
        array.append([])
        for j in range(0, length):
            if i == j:
                array[i].append(np.inf)
            else:
                array[i].append(cities[i].costTo(cities[j]))
    return np.array(array)


def branch_and_bound_reduce_array(array, lower_bound):
    """
    Given an array, reduce the values of the array so that there is at least one zero in every row and column. Add the
    reduced value to the lower bound if necessary.

    Time complexity:
    Iterate through every row in the matrix and obtain the minimum value. Then iterate through every column in the
    matrix and obtain the minimum value. Reduce the rows and columns by their minimum values respectively.
    Time: O(n^2)

    Space Complexity:
    Must contain the full n x n matrix to reduce.
    Space: O(n^2)

    """
    for i in range(0, len(array)):  # Iterate through every row O(n) rows.
        row = array[i, :]
        if np.isinf(row).all():
            # skip already explored rows
            continue
        min_val = np.min(row)  # Find the min in O(n) columns for total O(n^2)
        if min_val == np.inf:
            return None, np.inf
        if min_val > 0:
            lower_bound += min_val  # Reduce every value by the minimum in O(n) time.
            array[i, :] -= min_val

    for j in range(0, len(array)):  # Iterate through every column O(n) columns.
        col = array[:, j]
        if np.isinf(col).all():
            # skip already explored columns
            continue
        min_val = np.min(col)  # Find the min in O(n) rows for total O(n^2)
        if min_val == np.inf:
            return None, np.inf
        if min_val > 0:
            lower_bound += min_val  # Reduce every value by the minimum in O(n) time.
            array[:, j] -= min_val

    return array, lower_bound


def two_opt_total_distance(cities, order):
    """
    Takes an Array of cities and an array of the order of those cities,
    and sums the distance from the first city to the last along the order.
    """
    dist = 0
    for i in range(len(order)):
        j = i + 1
        if i == len(order) - 1:
            j = 0
        dist += cities[order[i]].costTo(cities[order[j]])
    return dist


def print_matrix(arr):
    # Determine the width of each column based on the maximum number of digits in any element
    col_widths = [max(len('{:.5f}'.format(x)) for x in col) for col in arr.T]

    # Print the array row by row, with values aligned within each column
    for row in arr:
        for i, x in enumerate(row):
            if x != math.inf and not math.isnan(x):
                print('{:{width}d}'.format(int(x), width=col_widths[i]), end=' ')
            else:
                # Use the appropriate column width and right-align the value
                print('{:{width}.5f}'.format(x, width=col_widths[i]), end=' ')
        print()  # Start a new row after all columns have been printed
    print()


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' 
    <summary>
    This is the entry point for the default solver
    which just finds a valid random tour.  Note this could be used to find your
    initial BSSF.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of solution,
    time spent to find solution, number of permutations tried during search, the
    solution found, and three null values for fields not used for this
    algorithm</returns>
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' 
    <summary>
    This is the entry point for the greedy solver, which you must implement for
    the group project (but it is probably a good idea to just do it for the branch-and
    bound project as a way to get your feet wet).  Note this could be used to find your
    initial BSSF.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution,
    time spent to find best solution, total number of solutions found, the best
    solution found, and three null values for fields not used for this
    algorithm</returns>
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        solution = None
        start_time = time.time()

        startCity = cities[0]

        solution = self.greedySolver(startCity)

        i = 1
        while solution == None and i < ncities - 1 and (time.time() - start_time < time_allowance):
            print("had to try again")  # fixme
            startCity = cities[i]
            solution = self.greedySolver(startCity)
            i += 1

        if solution == None:
            foundTour = False
        else:
            foundTour = True

        end_time = time.time()
        results['cost'] = solution.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = solution
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def find_nearest(self, city, visited):
        cities = self._scenario.getCities()
        nearest_city = None
        nearest_distance = sys.maxsize
        for c in cities:
            if c == city or c in visited:
                continue
            distance = city.costTo(c)
            if distance < nearest_distance:
                nearest_city = c
                nearest_distance = distance
        return nearest_city, nearest_distance

    def greedySolver(self, startCity):
        cities = self._scenario.getCities()
        visited = []
        current_city = startCity
        visited.append(current_city)
        total_distance = 0
        while len(visited) < len(cities):
            nearest_city, distance = self.find_nearest(current_city, visited)
            if nearest_city is None:
                return None
            visited.append(nearest_city)
            total_distance += distance
            current_city = nearest_city
        total_distance += current_city.costTo(cities[0])

        return TSPSolution(visited)

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        """
        This function is very complex, so bare with me. This is the branch and bound algorithm that initializes and runs
        the branch and bound to find an optimal solution to TSP. This uses a heap queue from python to take the state
        with the lowest bound and continue to dig deeper until it finds a possible solution. This implementation will
        prioritize the lowest depth in the tree before the lowest bound in order to update the BSSF as fast as possible
        and to begin to prune the search tree.

        Time Complexity:
        The branch and bound algorithm will create n - 1 branches which could each have n - 1 branches to be evaluated.
        As we go down the tree, and we attempt to prune these branches, the time complexity on average is far better
        than this worst case scenario, but the total time complexity could be O(n!) if every branch must be evaluated to
        find the optimal solution.
        Time: Worst case: O(n!) with average case O(n^2 * 2^n)

        Space Complexity:
        We must store the total path, reduced array matrix and lower bound for every branch in the tree. Because there
        are a possible total n branches on the heap queue at a single time, this means we could have n * n^2 space.
        Space: O(n^3)
        """
        # Get Default tour for bssf to start pruning
        default = self.defaultRandomTour()
        bssf_cost = default['cost']
        if bssf_cost == math.inf:
            return default

        # initialize values
        best_path = []
        bssf = []
        cities = self._scenario.getCities()
        ncities = len(cities)
        queue = []
        this_path = [0]
        array = branch_and_bound_initialize_array(cities)
        lower_bound = 0
        array, lower_bound = branch_and_bound_reduce_array(array, lower_bound)  # One reduce of O(n^2)

        # push first value onto heap
        heapq.heapify(queue)
        heapq.heappush(queue, (ncities - 1, lower_bound, this_path, array, 0, 1))

        # initialize result values
        bssf_updates = 0
        total = 1
        pruned = 0
        max_q = 1
        start_time = time.time()
        while queue and time.time() - start_time < time_allowance:  # This loop is majority of the time: O(n^2 * 2^n)
            depth_remaining, true_lb, path, this_array, current_city, depth = heapq.heappop(queue)
            # Update max queue size
            if len(queue) > max_q:
                max_q = len(queue)

            # prune branches that are too large to reduce search tree
            if true_lb >= bssf_cost:
                pruned += 1
                continue

            # If this is a valid cycle, check if bssf should be updated
            if depth_remaining == 0:
                if true_lb <= bssf_cost:
                    bssf_cost = true_lb
                    best_path = path
                    bssf_updates += 1
                continue

            # explore unvisited cities
            for next_city in range(ncities):  # Check n - 1 branches to see if they should be added to the queue O(2^n)
                if next_city in path:
                    continue
                travel_cost = this_array[current_city][next_city]
                if travel_cost == np.inf:
                    continue
                if true_lb + travel_cost >= bssf_cost:
                    continue
                new_array = this_array.copy()
                new_array[current_city, :] = np.inf
                new_array[:, next_city] = np.inf
                new_array[next_city, current_city] = np.inf
                new_array, new_lb = branch_and_bound_reduce_array(new_array, true_lb)  # Reduce each state in n^2 time so O(2^n * n^2)
                new_lb = new_lb + travel_cost
                new_depth = depth + 1
                new_depth_remaining = ncities - new_depth
                # push state onto queue
                if new_lb != np.inf and new_lb <= bssf_cost:
                    new_path = path.copy()
                    new_path.append(next_city)
                    heapq.heappush(queue, (new_depth_remaining, new_lb, new_path, new_array, next_city, new_depth))
                else:
                    pruned += 1
                total += 1
        end_time = time.time()

        # Return the best solution found
        if best_path:
            for i in range(len(best_path)):
                bssf.append(cities[best_path[i]])
            bssf = TSPSolution(bssf)
        else:
            bssf = default['soln']
            print("Using default")

        results = {'cost': bssf_cost, 'time': end_time - start_time, 'count': bssf_updates,
                   'soln': bssf, 'max': max_q, 'total': total, 'pruned': pruned}
        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy(self, time_allowance=60.0):
        """Solve the TSP using the 2-opt algorithm."""
        cities = self._scenario.getCities()
        greedy_start = self.greedy()
        order = []
        for city in greedy_start['soln'].route:
            order.append(cities.index(city))

        best_distance = two_opt_total_distance(cities, order)
        start_time = time.time()
        found_improvement = True
        bssf_updates = 0

        while found_improvement and time.time() - start_time < time_allowance: #O(n^5), space = O(n)
            found_improvement = False
            for i in range(0, len(order) - 1):
                for j in range(i+1, len(order)):
                    # Swap the cities
                    new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    # Calculate the new distance
                    new_distance = two_opt_total_distance(cities, new_order)
                    # If the new order is better, update the current order and best distance
                    if new_distance < best_distance:
                        order = new_order
                        best_distance = new_distance
                        found_improvement = True
        end_time = time.time()
        final_route = []
        for index in order:
            final_route.append(cities[index])
        bssf = TSPSolution(final_route)
        bssf_cost = bssf.cost

        results = {'cost': bssf_cost, 'time': end_time - start_time, 'count': bssf_updates,
                   'soln': bssf, 'max': None, 'total': None, 'pruned': None}
        return results
