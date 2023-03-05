import pandas as pd
import numpy as np
import cvxpy
import scipy
import cvxopt


def double_HP(target: pd.Series, lambda_value: float):
    y = target.to_numpy()
    y = np.log(y)
    n = y.size
    ones_row = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)

    solver = cvxpy.CVXOPT
    reg_norm = 2

    x = cvxpy.Variable(shape=n)
    objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x)
                    + lambda_value * cvxpy.norm(D@x, reg_norm))
    problem = cvxpy.Problem(objective)
    problem.solve(solver=solver, verbose=False)
    np.array(x.value)

    return np.array(x.value)


