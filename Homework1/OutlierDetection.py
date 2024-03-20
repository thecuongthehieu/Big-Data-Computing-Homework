from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import heapq
import math
import timeit

# Task 1
def ExactOutliers(points, D, M, K):
	outliers_count = 0
	outliers = []
	for i in range(len(points)):
		count = 0
		for j in range(len(points)):
			d = euclidean_distance(points[i], points[j])
			if (d <= D):
				count += 1

		if (count <= M):
			outliers_count += 1
			heapq.heappush(outliers, (-count, points[i])) # (min-heap with priority = -count) == (max-heap with priority = count)
			
			# K outliers
			if (len(outliers) > K):
				heapq.heappop(outliers)
	
	# 1. Print the number of outliers
	print("Number of Outliers =", outliers_count)

	# 2. Print first K outliers
	sorted_outliers = [] # non-decreasing order of -count
	while len(outliers) > 0:
		sorted_outliers.append(heapq.heappop(outliers))

	# print in non-decreasing order of count
	for i in range(len(sorted_outliers) - 1, -1, -1):
		print("Point: ({},{})".format(sorted_outliers[i][1][0], sorted_outliers[i][1][1]))

# Task 2 
def MRApproxOutliers(inputPoints, D, M, K):
	# Lambda
	ld = D / (2 * math.sqrt(2))

	# Step A
	cells = (inputPoints.map(lambda point: get_cell_id(point, ld))
		  .mapPartitions(count_point_per_cell_per_partition)
		  .reduceByKey(lambda x, y: x + y))
	
	# for cell in cells.collect():
	# 	print(cell)
	
	# Step B
	cells = (cells.map(lambda elem: (1, elem))
		  .groupByKey()
		  .flatMap(compute_full_info_per_cell))

	# Sure Outliers: N7(C) <= M
	# Uncertain Points: N3(C) <= M and N7(C) > M

	# 1. Print the number of sure outliers
	print("Number of sure outliers =", cells.map(lambda elem: collect_sure_outliers(elem, M)).reduce(lambda x, y: x + y))

	# 2. Print the number of uncertain points
	print("Number of uncertain points =", cells.map(lambda elem: collect_uncertain_points(elem, M)).reduce(lambda x, y: x + y))

	# 3. Print first K non-empty cells in non-dereasing order of N3(C)
	sorted_cells = cells.map(collect_size_and_identifier_of_cells).sortByKey().take(K)
	for cell in sorted_cells:
		print("Cell: ({},{})  Size = {}".format(cell[1][0], cell[1][1],cell[0]))

def euclidean_distance(p, q):
	return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def extract_point(line):
	# TODO: handle line = empty '' 
	line = line.strip('\n')	
	pair = line.split(',')
	return (float(pair[0]), float(pair[1])) # tuple

def get_cell_id(point, ld):
	return (math.floor(point[0] / ld), math.floor(point[1] / ld))

def count_point_per_cell_per_partition(cells):
	cells_dict = {}
	for cell in cells:
		if cell not in cells_dict.keys():
			cells_dict[cell] = 1
		else:
			cells_dict[cell] += 1
	return [(key, cells_dict[key]) for key in cells_dict.keys()]

# Compute N3(C) and N7(C)
def compute_full_info_per_cell(elem):
	cell_infos = elem[1]
	cell_size_dict = {} # For quickly look-up based on key
	n_three_dict = {}
	n_seven_dict = {}

	for cell_info in cell_infos:
		cell_size_dict[cell_info[0]] = cell_info[1]
		n_three_dict[cell_info[0]] = 0
		n_seven_dict[cell_info[0]] = 0


	for cell_info in cell_infos:
		cur_cell_id = cell_info[0]

		for i in range (-1, 2, 1):
			for j in range (-1, 2, 1):
				neighbor_cell_id = (cur_cell_id[0] + i, cur_cell_id[1] + j) 
				if neighbor_cell_id in cell_size_dict.keys():
					n_three_dict[cur_cell_id] += cell_size_dict[neighbor_cell_id]	

		for i in range (-3, 4, 1):
			for j in range (-3, 4, 1):
				neighbor_cell_id = (cur_cell_id[0] + i, cur_cell_id[1] + j) 
				if neighbor_cell_id in cell_size_dict.keys():
					n_seven_dict[cur_cell_id] += cell_size_dict[neighbor_cell_id]

	#  (key, [count_per_cell, n_three_per_cell, n_seven_per_cell])
	return [(key, [cell_size_dict[key], n_three_dict[key], n_seven_dict[key]]) for key in cell_size_dict.keys()]		

# Emit count of Sure outliers: N(7) <= M
def collect_sure_outliers(elem, M):
	cell_info = elem[1]
	cell_size = cell_info[0]
	n_seven_count = cell_info[2]

	if n_seven_count <= M:
		return cell_size
	else:
		return 0

# Emit count of Uncertain Points: N3(C) <= M and N7(C) > M
def collect_uncertain_points(elem, M):
	cell_info = elem[1]
	cell_size = cell_info[0]
	n_three_count = cell_info[1]
	n_seven_count = cell_info[2]

	if n_three_count <= M and n_seven_count > M:
		return cell_size
	else:
		return 0

# Emit size and identifier of cell
def collect_size_and_identifier_of_cells(elem):
	cell_id = elem[0]
	cell_info = elem[1]
	cell_size = cell_info[0]
	
	return((cell_size, cell_id))

# For Debugging
def identity_fun(x):
	print(x)
	return x

# Helper function
def parse_int_arg(argv):
	parts = argv.split('=')
	return int(parts[1])

# Helper function
def parse_float_arg(argv):
	parts = argv.split('=')
	return float(parts[1])

def main():
	# Check the number of args
	assert len(sys.argv) == 6, "Usage: python OutlierDetection.py <file_name> D=<D> M=<M> K=<K> L=<L>"

	# Spark setup 
	conf = SparkConf().setAppName('OutlierDetection')
	sc = SparkContext(conf=conf)

	# Print args
	print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

	# Read input file
	data_path = sys.argv[1]
	assert os.path.isfile(data_path), "File or folder not found"

	# Read D, M, K, L
	D = parse_float_arg(sys.argv[2])
	M = parse_int_arg(sys.argv[3])
	K = parse_int_arg(sys.argv[4])
	L = parse_int_arg(sys.argv[5])

	###########################################################################################
	###########################################################################################

	# Preprocess data
	rawData = sc.textFile(data_path,minPartitions=K).cache()
	inputPoints = rawData.map(lambda line: extract_point(line)).repartition(numPartitions=L)
	
	# Prints the total number of points.
	totalPoints = inputPoints.count()
	print("Number of points =", totalPoints)

	if (totalPoints <= 200000): 
		# Task 1: ExactOutliers	
		listOfPoints = inputPoints.collect()
		start_ts = timeit.default_timer()
		ExactOutliers(listOfPoints, D, M, K)
		elapsed_time = timeit.default_timer() - start_ts
		print("Running time of ExactOutliers = {} ms".format(math.ceil(elapsed_time * 1000)))

	# Task 2: MRApproxOutliers
	start_ts = timeit.default_timer()
	MRApproxOutliers(inputPoints, D, M, K)	
	elapsed_time = timeit.default_timer() - start_ts
	print("Running time of MRApproxOutliers = {} ms".format(math.ceil(elapsed_time * 1000)))

if __name__ == "__main__":
	main()