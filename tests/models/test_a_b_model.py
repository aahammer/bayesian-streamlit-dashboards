import pytest

from models.a_b_model import create, evaluate_variants, Effect
from models.types import BetaPrior

import pymc
import numpy
import math


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


@pytest.mark.parametrize('control, variants, effects', [
     [
         [0]*10,
         [[0]*5+[1]*5],
         [{'lift':0.5, 'confidence':0.5}]
     ],
    [
        [0.1] * 7 + [0.2] * 3,
        [[0.15] * 10],
        [{'lift': 0.02, 'confidence': 0.7}]
    ],

])
def test_evaluate_variants(control, variants, effects):

    control = numpy.array(control)
    variants = numpy.array(variants)

    results = evaluate_variants(control, variants)


    for i, e in enumerate(effects):
         assert(math.isclose(results[i].lift, e['lift']))
         assert(math.isclose(results[i].confidence, e['confidence']))
