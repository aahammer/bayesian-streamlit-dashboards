import pymc

from typing import List
from .types import BetaPrior

import numpy
from numpy.typing import NDArray

from pydantic import BaseModel, confloat

class Effect(BaseModel):
    lift: confloat()
    confidence: confloat(ge=0, le=1)


def create(priors: List[BetaPrior]) -> pymc.Model:

    model = pymc.Model()

    number_of_variants = len(priors)

    alphas = [None] * number_of_variants
    betas  = [None] * number_of_variants

    for i, prior in enumerate(priors):
        alphas[i] = prior.alpha
        betas[i] = prior.beta

    with model:
        p = pymc.Beta('p', alpha=alphas, beta=betas, shape=number_of_variants)

    return model

def evaluate_variants(control: NDArray[numpy.float64], variants: NDArray[numpy.float64] ) -> List[Effect]:

    effects = []
    for variant in variants:
        lift = variant.mean() - control.mean()
        confidence = numpy.array([variant - control > 0]).mean()
        effects.append(Effect(**{'lift': lift, 'confidence': confidence}))

    return effects


