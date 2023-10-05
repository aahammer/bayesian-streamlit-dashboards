from typing import List, Tuple
from pydantic import BaseModel, conint
import pymc

from .types import BetaPrior

class StepStatus(BaseModel):
    """
    Represents the status of a particular step in the funnel.
    """
    active: conint(ge=0)
    dropped: conint(ge=0)


def create(steps: List[str], priors: List[BetaPrior], prefill: List[int] = None) -> pymc.Model:
    """
    Creates a probabilistic model based on provided steps and conversion rate priors.

    Args:
    - steps: List of funnel steps.
    - priors: List of beta priors corresponding to each step (alpha and beta values)
    - prefill: List of initial values for each step.

    Returns:
    A PyMC model object.
    """

    if prefill is None:
        prefill = [0] * len(steps)

    assert len(steps) == len(priors) + 1
    assert len(prefill) == len(steps)

    # Validate the types of items in priors
    for prior in priors:
        assert isinstance(prior, BetaPrior), "All priors should be instances of the BetaPrior class"

    model = pymc.Model()

    with model:
        betas = [None] * len(priors)
        binomials = [None] * len(steps)
        active = [None] * len(steps)

        active[0] = pymc.Deterministic(f'Active in {steps[0]}', pymc.MutableData('', prefill[0]))

        for step, step_name in enumerate(steps[:-1]):
            betas[step] = pymc.Beta(f'% {step_name} -> {steps[step + 1]}', priors[step].alpha, priors[step].beta)

            retained = binomials[step - 1] if step >= 1 else 0
            binomials[step] = pymc.Binomial(f'Number Retained from {step_name} to {steps[step + 1]}',
                                            p=betas[step], n=prefill[step] + retained)
            active[step + 1] = pymc.Deterministic(f'{steps[step + 1]}', prefill[step + 1] + binomials[step])

    return model


def update_priors(priors: List[BetaPrior], statuses: List[StepStatus]) -> List[BetaPrior]:
    """
    Updates the priors based on the provided step statuses.

    Args:
    - priors: List of initial beta priors.
    - statuses: List of statuses for each step.

    Returns:
    List of updated beta priors.
    """

    return [
        BetaPrior(
            alpha=p.alpha + sum(status.active + status.dropped for status in statuses[i + 1:]),
            beta=p.beta + statuses[i].dropped
        )
        for i, p in enumerate(priors)
    ]
