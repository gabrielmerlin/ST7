import numpy as np
import cvxpy as cp

wrob = np.array([0.18, 0.23, 0.12, 0.25])
lan = 4
k = 0.2
w = cp.Variable(4)
sigma = np.array([[0.04, 0.024, 0.01, 0.0], [0.024, 0.0225, 0.006, 0.0015], [0.01, 0.006, 0.01, 0.0025], [0.0, 0.0015, 0.0025, 0.0025]])


#print(invsig)
omega = np.zeros((4,4))
for i in range(4):
    omega[i][i]=sigma[i][i]
A, B = np.linalg.eig(omega)
print(A)
d = np.dot(wrob, omega)
dprime = np.dot(d, np.transpose(wrob))
dprime = k * dprime
d = dprime * d
e = np.dot(wrob, sigma)
mu = np.array([0.02784383, 0.02118275, 0.00785006, 0.00254033])


risk = cp.quad_form(w, omega)
risk = cp.multiply(lan/2, risk)



#error_carre = cp.quad_form(w, omega)
#error = cp.multiply(k, cp.sqrt(error_carre))
error = cp.norm(np.linalg.cholesky(omega) * w, 2)   # √ w.t * omega * w
error=cp.multiply(k,error)
objective = cp.Maximize((mu * w) - risk - error)
constraints = [w >= 0, cp.sum(w) == 1]
prob = cp.Problem(objective, constraints)
prob.solve()
print(w.value)
#def forme (x,y)=

def optimisation_rob(mu_sigma_dict,lan,k):
    w_d={}
    for date in mu_sigma_dict:
        mu,sigma=mu_sigma_dict[date]
        mu,sigma=mu.to_numpy(),sigma.to_numpy()
        for i in range(len(sigma)):
            omega[i][i]=sigma[i][i]
        risk = cp.quad_form(w, omega)
        risk = cp.multiply(lan/2, risk)
        error = cp.norm(np.linalg.cholesky(omega) * w, 2)   # √ w.t * omega * w
        objective = cp.Maximize((mu * w) - risk - error)
        constraints = [w >= 0, cp.sum(w) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        w_d[date] = pd.Series(w.value, index=mu.index)
    return pd.DataFrame(w_d).stack().rename('Poids', axis='column')
print((w.value))
