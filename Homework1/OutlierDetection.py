from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import heapq
import math

# TODO: What about overflow of memory?
def ExactOutliers(points, D, M, K):
	# For debugging
	for i in range(len(points)):
		print(points[i])
	print()

	num_outliers = 0
	outliers = []
	for i in range(len(points)):
		count = 0
		for j in range(len(points)):
			if i != j:
				d = dist(points[i], points[j])
				if (d <= D):
					count += 1

		if (count <= M):
			num_outliers += 1
			heapq.heappush(outliers, (-count, points[i])) # (min-heap with priority = -count) == (max-heap with priority = count)
			if (len(outliers) > K):
				heapq.heappop(outliers)

	
	# Log 		
	print(num_outliers)

	sorted_outliers = [] # non-decreasing order of -count
	while len(outliers) > 0:
		sorted_outliers.append(heapq.heappop(outliers))
	
	# print in non-decreasing order of count
	for i in range(len(sorted_outliers) - 1, 0, -1):
		# TODO: print only points
		print(sorted_outliers[i])

# TODO: Finish
def MRApproxOutliers(inputPoints, D, M, K):
	ld = D / (2 * math.sqrt(2))

	# Step A
	inputPoints.map(lambda point: get_cell_id(point)).flatMap()

	# Step B


def dist(p, q):
	return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def get_points(data_path):
	file = open(data_path, 'r')
	points = []
	while True:
		line = file.readline()
		if not line:
			break

		line = line.strip('\n')
		if (line == ''):
			continue

		pair = line.split(',')
		points.append((float(pair[0]), float(pair[1])))
 
	return points

def extract_point(line):
	# TODO: handle line = empty '' 
	line = line.strip('\n')	
	pair = line.split(',')
	return (float(pair[0]), float(pair[1])) # tuple

def get_cell_id(point, ld):
	return ((int) (point[0] / ld), (int) (point[1] / ld))

def count_point_per_cell_per_partition(cells):
	cells_dict = {}
	for cell in cells:
		if cell not in cells_dict.keys():
			cells_dict[cell] = 1
		else:
			cells_dict[cell] += 1
	return [(key, cells_dict[key]) for key in cells_dict.keys()]

# Compute N3(C)
def compute_n_three_per_cell_per_partition(pointcounts):
	n_three_list = {}
	pointcounts_dict = {}
	for pointcount in pointcounts:
		pointcounts_dict[pointcount[0]] = pointcount[1]

	for pointcount in pointcounts:
		cell = pointcount[0]
		n_three_list[cell] = 0

		for i in range (-1, 2, 1):
			for j in range (-1, 2, 1):
				neighbor_cell = (cell[0] + i, cell[1] + j) 
				if neighbor_cell in pointcounts_dict.keys():
					n_three_list[cell] += pointcounts_dict[neighbor_cell]

	return [(key, n_three_list[key]) for key in n_three_list.keys()]	


# Compute N7(C)
def compute_n_seven_per_cell_per_partition(pointcounts):
	n_seven_list = {}
	pointcounts_dict = {}
	for pointcount in pointcounts:
		pointcounts_dict[pointcount[0]] = pointcount[1]

	for pointcount in pointcounts:
		cell = pointcount[0]
		n_seven_list[cell] = 0

		for i in range (-3, 4, 1):
			for j in range (-3, 4, 1):
				neighbor_cell = (cell[0] + i, cell[1] + j) 
				if neighbor_cell in pointcounts_dict.keys():
					n_seven_list[cell] += pointcounts_dict[neighbor_cell]

	#  (key, [count_per_cell, n_seven_per_cell])
	return [(key, [pointcounts_dict[key], n_seven_list[key]]) for key in n_seven_list.keys()]	

# Compute sure outliers
def compute_sure_outliers(cell_info_list, M):
	point_count = 0
	n_seven_count = 0
	for cell_info in cell_info_list[1]:
		point_count += cell_info[0]
		n_seven_count += cell_info[1]

	if n_seven_count < M:
		return (cell_info_list[0], (point_count, n_seven_count))
	else:
		return (cell_info_list[0], (0, n_seven_count)) 

# Demo for print outliers
def demo_fun(inputPoints, D, M, K):
	ld = D / (2 * math.sqrt(2))
	cells = (inputPoints.map(lambda point: get_cell_id(point, ld))
		  .mapPartitions(count_point_per_cell_per_partition)
		  .mapPartitions(compute_n_seven_per_cell_per_partition)
		  .groupByKey()
		  .map(lambda cell_info_list : compute_sure_outliers(cell_info_list, M)))
	
	print(cells.map(identity_fun).count())



# For debugging
def identity_fun(x):
	print(x)
	return x

def main():
	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 6, "Usage: python OutlierDetection.py <D> <M> <K> <L> <file_name>"


	# INPUT READING
	# TODO: assert values

	# 1. Read D
	D = sys.argv[1]
	D = float(D)
	
    # 2. Read M
	M = sys.argv[2]
	assert M.isdigit(), "M must be an integer"
	M = int(M)

    # 3. Read K    
	K = sys.argv[3]
	K = int(K)
	
    # 4. Read L
	L = sys.argv[4]
	L = int(L)
	
    # 5. Read input file
	data_path = sys.argv[5]
	assert os.path.isfile(data_path), "File or folder not found"

	# Get input
	points = get_points(data_path)

	# Print
	print()
	print()
	print()

	# 1) ExactOutliers	
	# MRApproxOutliers(points, D, M, K)

	# 2) MRApproxOutliers

	# Spark setup 
	conf = SparkConf().setAppName('OutlierDetection')
	sc = SparkContext(conf=conf)

	# Preprocess data
	rawData = sc.textFile(data_path,minPartitions=K).cache()
	inputPoints = rawData.map(lambda line: extract_point(line)).repartition(numPartitions=1)

	###########################

	ld = D / (2 * math.sqrt(2))

	# Step A
	pointCounts = inputPoints.map(lambda point: get_cell_id(point, ld)).mapPartitions(count_point_per_cell_per_partition)
	print(pointCounts.map(identity_fun).count())

	# Step B	
	## Sure Non-outliers: N3(C) > M 
	## Sure Outliers: N7(C) <= M
	## Uncertain Points: N3(C) <= M and N7(C) > M

	# cell_infos = pointCounts.mapPartitions(compute_n_seven_per_cell_per_partition).groupByKey()
	# sure_outlier_count = cell_infos.map(lambda cell_info_list : compute_sure_outliers(cell_info_list, M))
	# sure_outlier_count.map(identity_fun).count()
	# print(sure_outlier_count)

	demo_fun(inputPoints, D, M, K)

if __name__ == "__main__":
	main()