import numpy
import cvxpy as cp
w= cp.Variable(4)
print(w)
SD=numpy.array([[0.15],[0.2],[0.09],[0.003]])
SR=numpy.array([[0.4],[0.35],[0.45],[0.25]])
rho=numpy.array([[1.,0.8,0.7,0.5],[0.8,1,0.2,0],[0.7,0.2,1,0.8],[0.5,0,0.8,1]]
