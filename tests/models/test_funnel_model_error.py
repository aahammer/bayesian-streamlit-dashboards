import pytest
from pydantic import ValidationError

# system under test
from models.funnel_model import create
from models.types import BetaPrior

def test_create_error_to_many_steps():
    steps = ['A', 'B', 'C', 'D']
    priors = [BetaPrior(**{'alpha': 3, 'beta': 3}), BetaPrior(**{'alpha': 2, 'beta': 2})]

    with pytest.raises(AssertionError):
        create(steps, priors)

def test_create_error_to_negative_alpha_beta():

    with pytest.raises(ValidationError):
        BetaPrior(**{'alpha': -1, 'beta': 3})

