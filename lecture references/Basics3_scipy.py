from scipy.sparse import *
from numpy import *

M = array([[1, 0, 0, 0], [0, 3, 0, 0], [0, 4, 1, 0], [1, 0, 0, 5]])

A = coo_matrix(M)

print("COO Coordinate Format")
print(A.row)
print(A.col)
print(A.data)

A_csr = csr_matrix(A)

print("CSR Compressed Row Format, fast row access")
print(A_csr.indices)
print(A_csr.indptr)
print(A_csr.data)

A_csc = csc_matrix(A)

print("CSC Compressed Column Format, fast column access")
print(A_csc.indices)
print(A_csc.indptr)
print(A_csc.data)