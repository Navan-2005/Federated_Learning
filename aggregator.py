def fedavg(updates):
    """Federated Averaging.

    updates: list of (state_dict, num_samples). Returns the sample-weighted
    average of each parameter: sum(n_k / n_total * w_k).
    """
    total = sum(n for _, n in updates)
    averaged = {}
    for key in updates[0][0]:
        averaged[key] = sum(
            state[key].float() * (n / total) for state, n in updates
        ).to(updates[0][0][key].dtype)
    return averaged
