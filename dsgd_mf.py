#******************************************************
# HW7: Distributed SGD for Matrix Factorization on Spark
# Name: Shiny Singh
# Andrew ID: shinys
#******************************************************
# The objective is to factorize a matrix V into WH, where W is a MXr matrix and H is an rxN matrix
# The input is a CSV file containing triplets of the form (i,j,V(i,j)) where V is the input matrix 
# that needs to be factorized
# M : Number of Users
# N : Number of Movies
# r : Number of Matrix Factors 
# lambda: Regularization Parameter of L-2 Regularization
# The final result is stored in w.csv and h.csv files
#******************************************************

from pyspark import SparkContext
from operator import itemgetter
import sys
import numpy
import scipy
from scipy import sparse
import csv
#Function for splitting every line into triplets (i,j,V(i,j))
def split_input(line):
	return map(int,line.split(','))

#Function for obtaining the strata number for a point (i,j)
def get_strata(row, col, num_row,num_col):
	if num_row>=num_col:
		return (row-col)%num_row
	else:
		return (col - row)%num_col

#Function for initializing one row of the W matrix as a 1xr array
def initializeW(x,num_factors):
	return numpy.random.rand(num_factors)

#Function for initializing one column of the H matrix as a 1xr array
def initializeH(x,num_factors):
	return (numpy.random.rand(num_factors))

#Function for updating one row of the W matrix and returning it
def sgd_fun_W(W,H,x,r,l,n,beta,tao,M):
	product = W.dot(H)[0,0]
	Ni = len(W.nonzero()[0])
	epsilon = pow(n+tao,-beta)
	data = 2*epsilon*(x[2] - product)*H.transpose() 
	data = data - 2*epsilon*l*W*(1/Ni)
	data = data.toarray()
	return [x[0],data[0]]
# Function for updating one column of the H matrix and returning it
def sgd_fun_H(W,H,x,r,l,n,beta,tao,N):
	product = W.dot(H)[0,0]
	Nj = len(H.nonzero()[0])
	epsilon = pow(n+tao,-beta)
	data = 2*epsilon*(x[2] - product)*W - 2*epsilon*l*H.transpose()*(1/Nj)
	data = data.toarray()
	return [x[1],data[0]]

#******************* Main program starts ************************

# Initializing a Spark Context object
sc = SparkContext(appName = 'Matrix Factorization')
if len(sys.argv) < 9:
	print 'Insufficient arguments'

#Reading command line arguments
num_factors =int(sys.argv[1])
num_workers = int(sys.argv[2])
num_iterations = int(sys.argv[3])
beta = float(sys.argv[4])
lambda_value = float(sys.argv[5])
input_filepath = sys.argv[6]
outputW_file = sys.argv[7]
outputH_file = sys.argv[8]
tao = 100

#Converting the input CSV file to RDD format
input_file = sc.textFile(input_filepath)
#Splitting every input line into corresponding triplet
input_triplets = input_file.map(split_input)

# Finding the value of M and N from the obtaining triplets
M = input_triplets.map(lambda x: x[0]).reduce(lambda x,y: max(x,y))
N = input_triplets.map(lambda x: x[1]).reduce(lambda x,y: max(x,y))

# Finding the number of strata
B = input_triplets.map(lambda x: get_strata(x[0],x[1],M,N)).reduce(lambda x,y: max(x,y))

#Initializing W and H as random sparse matrices
W = scipy.sparse.rand(M,num_factors,density = 1)
H = scipy.sparse.rand(num_factors,N,density = 1)

for i in range(num_iterations):
	for b in range(B+1):
		num_updates = i*B + b
		# Filtering the triplets belonging to strata b
		blocks = input_triplets.filter(lambda x: get_strata(x[0],x[1],M,N)==b)
		# Updating W for all elements belonging to strata b
		W_temp = blocks.map(lambda x: (sgd_fun_W(W.getrow(x[0]-1), H.getcol(x[1]-1),x,num_factors,lambda_value,num_updates,beta,tao,M)))
		W_temp = W_temp.collect()
		W_temp = sorted(W_temp, key=itemgetter(0))
		for k in range(len(W_temp)):
			row_idx = W_temp[k][0]
			data = W_temp[k][1]
			col = [j for j in range(num_factors)]
			row = [row_idx-1 for j in range(num_factors)]
			updated_matrix = scipy.sparse.csr_matrix((data, (row, col)),shape = (M,num_factors))
			W = W + updated_matrix

		#Updating H for all elements belonging to strata b
		H_temp = blocks.map(lambda x: (sgd_fun_H(W.getrow(x[0]-1), H.getcol(x[1]-1),x,num_factors,lambda_value,num_updates,beta,tao,M)))
		H_temp = H_temp.collect()
		H_temp = sorted(H_temp, key=itemgetter(0))
		for k in range(len(H_temp)):
			col_idx = H_temp[k][0]
			data = H_temp[k][1]
			col = [col_idx-1 for j in range(num_factors)]
			row = [j for j in range(num_factors)]
			updated_matrix = scipy.sparse.csr_matrix((data, (row, col)),shape = (num_factors,N))
			H = H + updated_matrix

#Saving W and H matrices to CSV files
numpy.savetxt("w.csv", W.toarray(), delimiter=",")
numpy.savetxt("h.csv", H.toarray(), delimiter=",")
