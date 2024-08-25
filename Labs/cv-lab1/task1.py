import numpy as np

A = np.random.rand(200,10)
mu = np.zeros(A.shape[1])
for i in range(A.shape[0]):
    mu += A[i]
mu /= A.shape[0]
B = np.zeros_like(A)
for i in range(A.shape[0]):
     B[i] = A[i] - mu
B=A-(A.sum(axis=0)/A.shape[0])
print(B)
print("*****************************")
B=A-(A.mean(axis=0))
print(B)