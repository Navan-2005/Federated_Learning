def async_merge(global_state, client_state, base_alpha, staleness):
    """Staleness-weighted mixing: stale updates get a smaller mixing weight.

    alpha = base_alpha / (1 + staleness)
    global = (1 - alpha) * global + alpha * client
    """
    alpha = base_alpha / (1 + staleness)
    merged = {}
    for key, g in global_state.items():
        if g.is_floating_point():
            merged[key] = (1 - alpha) * g + alpha * client_state[key].to(g.dtype)
        else:
            merged[key] = g
    return merged, alpha


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
