import pytest

# system under test
from models.funnel_model import create, update_priors, StepStatus
from models.types import BetaPrior

# support libraries
import pymc


def test_create():
    steps = ['A', 'B', 'C']
    priors = [BetaPrior(**{'alpha': 3, 'beta': 3}), BetaPrior(**{'alpha': 2, 'beta': 2})]

    model = create(steps, priors)

    assert isinstance(model, pymc.Model)
    assert len([rv for rv in model.free_RVs if rv.owner.op.name == 'beta']) == len(priors)
    assert len([rv for rv in model.free_RVs if rv.owner.op.name == 'binomial']) == len(priors)


def test_create_with_actives():
    steps = ['A', 'B', 'C']
    priors = [BetaPrior(**{'alpha': 3, 'beta': 3}), BetaPrior(**{'alpha': 2, 'beta': 2})]

    active = [10, 1, 0]

    model = create(steps, priors, active)

    assert isinstance(model, pymc.Model)
    assert len([rv for rv in model.free_RVs if rv.owner.op.name == 'beta']) == len(priors)
    assert len([rv for rv in model.free_RVs if rv.owner.op.name == 'binomial']) == len(priors)


@pytest.mark.parametrize('prior, status, new_prior', [
    ([{'alpha': 0, 'beta': 0}, {'alpha': 0, 'beta': 0}],
     [{'active': 1, 'dropped': 1}, {'active': 1, 'dropped': 1}],
     [{'alpha': 2, 'beta': 1}, {'alpha': 0, 'beta': 1}]
     ),
    ([{'alpha': 0, 'beta': 0}, {'alpha': 0, 'beta': 0}, {'alpha': 0, 'beta': 0}],
     [{'active': 1, 'dropped': 1}, {'active': 1, 'dropped': 1}, {'active': 1, 'dropped': 1}],
     [{'alpha': 4, 'beta': 1}, {'alpha': 2, 'beta': 1}, {'alpha': 0, 'beta': 1}]
     ),
    ([{'alpha': 0, 'beta': 0}],
     [{'active': 1, 'dropped': 1}],
     [{'alpha': 0, 'beta': 1}]
     ),
    ([{'alpha': 5, 'beta': 1}, {'alpha': 2, 'beta': 6}],
     [{'active': 10, 'dropped': 6}, {'active': 22, 'dropped': 35}],
     [{'alpha': 62, 'beta': 7}, {'alpha': 2, 'beta': 41}]
     )
])
def test_update_priors(prior, status, new_prior):

    prior = [BetaPrior(**p) for p in prior]
    tmp_prior = prior.copy()
    status = [StepStatus(**s) for s in status]
    new_prior = [BetaPrior(**np) for np in new_prior]

    assert new_prior == update_priors(prior, status)
    assert prior == tmp_prior
