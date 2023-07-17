from scipy.optimize import linear_sum_assignment
import numpy as np

cost_mat_ = np.random.randint(1, 20, [5, 7])
print(cost_mat_, '\n')

cost_mat = np.repeat(cost_mat_, 3, axis=0)
print(cost_mat, '\n')

index, target = linear_sum_assignment(cost_mat)
print(index, target, '\n')

cost_ass = cost_mat.copy()
for i, t in zip(index, target):
    cost_ass[i][t] = 0
print(cost_ass, '\n')

cost = cost_mat[index, target].sum()
print('sum:', cost)
