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


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
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


	''' <summary>
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

	def greedy( self,time_allowance=60.0 ):
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
			print("had to try again") # fixme
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

	def branchAndBound( self, time_allowance=60.0 ):
		pass

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		results = {}
		pass


	def printMatrix(self, arr):
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


