import pytest

from models.a_b_model import create
from models.types import BetaPrior

import pymc


@pytest.mark.parametrize('priors', [
     [BetaPrior(**{'alpha': 0, 'beta': 0})],
     [BetaPrior(**{'alpha': 0, 'beta': 0}), BetaPrior(**{'alpha': 0, 'beta': 0})] ,
     [BetaPrior(**{'alpha': 0, 'beta': 0})]*23,
     [BetaPrior(**{'alpha': 0, 'beta': 0})]*100
])
def test_create(priors):

    model = create(priors)

    assert isinstance(model, pymc.Model)

    # assert that the beta distribution was created with a shape of n
    assert [rv.type.shape[0] for rv in model.free_RVs if rv.owner.op.name == 'beta'][0] == len(priors)
