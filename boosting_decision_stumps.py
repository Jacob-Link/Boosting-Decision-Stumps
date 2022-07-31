"""
created to solve a question for specific data set.
see PDF to understand the question solved
"""

import numpy as np

data = np.array([[1, 1], [1, 3], [2, 2], [3, 1], [3, 3]])
labels = np.array([1, 1, -1, 1, 1])

m = len(data)  # number of points in data set
d_vec = np.array([1/m]*5)  # initialize weight vector given to each point

# create weak learners
wl_1 = lambda x: 1 if x[1] < 5 else -1 
wl_2 = lambda x: 1 if x[1] < 1.5 else -1
wl_3 = lambda x: 1 if x[1] > 2.5 else -1
wl_list = [wl_1, wl_2, wl_3]

iteration_weight = np.array([])  # store weights, needed for final hypothesis
iteration_wl = []  # store weak learner used for each iteration 

T = 3  # num of iterations
for t in range(T):
  wl = wl_list[int(t%len(wl_list))]  # invoke weak learner
  iteration_wl.append(wl)  # store the wl used for each iteration
  pred = np.array([wl(i) for i in data])
  epsilon = d_vec[pred != labels].sum()
  wt =  0.5 * np.log((1/epsilon) - 1)  # calc iteration weight
  iteration_weight = np.append(iteration_weight, wt)  # store the iteration weight
  
  print(f"t = {t+1}:\n\tHypothesis: {np.array([wl(x) for x in data])}")
  print(f"\tEpsilon: {round(epsilon, 2)}\n\tW: {round(wt, 2)}")
  
  # update D vec
  denominator = sum([d_vec[j]*np.exp(-1*wt*labels[j]*pred[j]) for j in range(m)])
  for i in range(m):
    numerator = d_vec[i]*np.exp(-1*wt*labels[i]*pred[i])
    d_vec[i] = (numerator/denominator)

# calc output after T iterations
output_hypothesis_val = [sum([iteration_weight[i]*iteration_wl[i](point) for i in range(T)]) for point in data]
print_val = [round(x, 2) for x in output_hypothesis_val]
output_hypothesis = [int(np.sign(val)) for val in output_hypothesis_val]
output_hypothesis = np.array(output_hypothesis)

print(f'\nOutput:')
print(f'\tValue each point      >>> {print_val}:')
print(f'\tOutput hypothesis     >>> {output_hypothesis}')
print(f'\tTrue label            >>> {labels}')
print(f'\tClassification error  >>> {round(((labels!=output_hypothesis).sum() / m)*100, 2)}% ')