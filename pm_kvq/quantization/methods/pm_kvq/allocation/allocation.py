import numpy as np
import cvxpy as cp
import torch


def allocate_memory_budget(fbit_choices=[8, 4, 2], k_sensitivity=None, v_sensitivity=None, memory_budget=32.0, hidden_size=1024, max_len=32768, save_path=None):
    sen = []
    choices = len(fbit_choices)
    n_layers = len(k_sensitivity[fbit_choices[0]])
    num_variables = int(n_layers * choices)
    for i in range(n_layers):
        for n_bits in fbit_choices:
            sen.append(k_sensitivity[n_bits][i] + v_sensitivity[n_bits][i])
    sen = np.array(sen)

    choices_item = np.repeat(np.array([fbit_choices]), n_layers, axis=0).reshape(-1)
    kv_mem_per_layer = choices_item * hidden_size * 2 * max_len / 8 / 1024 / 1024

    # Objective. Construct a CVXPY problem
    x = cp.Variable(num_variables, integer=True)
    objective = cp.Minimize(cp.sum(sen @ x))

    # Constrain 1, 0<=x<=1
    A1_1 = np.diag(np.ones(num_variables))
    A1_2 = -np.diag(np.ones(num_variables))

    B1_1 = np.ones(num_variables) * 1
    B1_2 = np.zeros(num_variables)

    A1 = np.vstack((A1_1, A1_2))
    B1 = np.concatenate((B1_1, B1_2))

    # Constrain 2, x1+x2+...+xn=1, split into >=1 and <=1
    A2_1 = np.zeros((n_layers, num_variables))
    for mark in range(n_layers):
        A2_1[mark, choices * mark : choices * mark + choices] = 1

    A2_2 = -A2_1

    B2_1 = np.ones(n_layers)
    B2_2 = -np.ones(n_layers)

    A2 = np.vstack((A2_1, A2_2))
    B2 = np.concatenate((B2_1, B2_2))

    # constrain 3, average bitwidth
    A3 = kv_mem_per_layer
    B3 = np.array([memory_budget])

    # Constrain merge
    A = np.vstack((A1, A2, A3))
    B = np.concatenate((B1, B2, B3))
    constraints = [A @ x <= B]

    # Define and solve the CVXPY problem. (solver=cp.ECOS_BB)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIPY)

    optimal_idx = x.value.reshape(-1, len(fbit_choices)).argmax(axis=1)
    optimal_bitwidth = choices_item.reshape(-1, len(fbit_choices))[np.arange(len(optimal_idx)), optimal_idx]
    optimal_budget = [kv_mem_per_layer[n_bits] for n_bits in optimal_idx]

    if save_path is not None:
        torch.save(optimal_budget, save_path)

    return optimal_bitwidth, optimal_budget
