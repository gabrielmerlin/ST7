import numpy
import cvxpy as cp
w= cp.Variable(4)
print(w)
SD=numpy.array([[0.2],[0.15],[0.05],[0.001]])
SR=numpy.array([[0.4],[0.35],[0.45],[0]])
rho=numpy.array([[1.,0.8,0.5,0],[0.8,1,0.2,0],[0.5,0.2,1,0],[0,0,0,1]])
mu=numpy.array([[0],[0],[0],[0]],float)
d=numpy.transpose(mu)
for i in range(4):
    mu[i][0]=SD[i][0]*SR[i][0]
    muprime=numpy.transpose(mu)
d=numpy.dot(SD,numpy.transpose(SD))
sigma=numpy.zeros((4,4))
for i in range(4):
    for(j) in range(4):
        sigma[i][j]=rho[i][j]*d[i][j]
print(sigma)
objective = cp.Maximize(muprime*w)
risk=cp.quad_form(w,sigma)
constraints = [risk<= 0.1**2]
prob = cp.Problem(objective, constraints)
prob.solve()
#The optimal value for x is stored in `x.value`.
print(w.value)

