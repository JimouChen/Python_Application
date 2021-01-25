from scipy import optimize
import numpy as np

if __name__ == '__main__':
    # 确定c,A,b,Aeq,beq
    c = np.array([2, 3, -5])
    A = np.array([[-2, 5, -1], [1, 3, 1]])
    B = np.array([-10, 12])
    Aeq = np.array([[1, 1, 1]])
    Beq = np.array([7])

    # 求解
    res = optimize.linprog(-c, A, B, Aeq, Beq)
    print(res)
