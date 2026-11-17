import os

import torch

SEED = 20
HERE = os.path.dirname(os.path.abspath(__file__))


def state_to_json(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}


def state_from_json(payload):
    return {k: torch.tensor(v) for k, v in payload.items()}
