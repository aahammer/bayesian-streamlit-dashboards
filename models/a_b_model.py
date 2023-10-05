import pymc

from typing import List
from .types import BetaPrior

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