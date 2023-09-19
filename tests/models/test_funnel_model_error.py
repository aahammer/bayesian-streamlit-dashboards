import pytest
from pydantic import ValidationError

# system under test
from models.funnel_model import create

# support libraries
import pymc

def test_create_error_to_many_steps():
    steps = ['A', 'B', 'C', 'D']
    priors = [{'alpha': 3, 'beta': 3}, {'alpha': 2, 'beta': 2}]

    with pytest.raises(AssertionError):
        create(steps, priors)

def test_create_error_to_negative_alpha_beta():
    steps = ['A', 'B', 'C']
    priors = [{'alpha': -1, 'beta': 3}, {'alpha': 2, 'beta': 2}]

    with pytest.raises(ValidationError):
        create(steps, priors)
