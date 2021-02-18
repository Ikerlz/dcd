import cvxpy as cp
import numpy as np


n = 100
A = np.random.randn(n, n)

# Construct the problem.
Y = cp.Variable((n, n))
S = cp.Variable((n, n))

objective = cp.Minimize(cp.normNuc(Y) + cp.abs(cp.atoms.sum(Y)) +  + _)
cp.norm()
constraints = [0 <= Y, Y <= 1, S + Y == A]
cp.atoms.sum(Y)


prob = cp.Problem(objective, constraints)
#
print("Optimal value", prob.solve())
print("Optimal var")
print(Y.value)  # A numpy ndarray# .
print(S.value)

# Problem data.
# m = 100
# n = 5
# numpy.random.seed(1)
# A = numpy.random.randn(m, n)
# b = numpy.random.randn(m)
#
# # Construct the problem.
# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A @ x - b))
# constraints = [0 <= x, x <= 1]
# prob = cp.Problem(objective, constraints)
#
# print("Optimal value", prob.solve())
# print("Optimal var")
# print(x.value)  # A numpy ndarray.
# cp.norm()