    from typing import List, Tuple
    from pydantic import BaseModel, confloat, conint

    import pymc

    class BetaPrior(BaseModel):
        alpha: confloat(ge=0)
        beta: confloat(ge=0)

    class StepStatus(BaseModel):
        active: conint(ge=0)
        dropped: conint(ge=0)


    def create(steps: List[str], priors: List[BetaPrior], prefill: List[int] = None) -> pymc.Model:

        if prefill is None:
            prefill = [0] * len(steps)

        assert len(steps) == len(priors) + 1
        assert len(prefill) == len(steps)
        [isinstance(prior,BetaPrior) for prior in priors]

        model = pymc.Model()

        with model:

            betas = [None] * len(priors)
            binomials = [None] * len(steps)
            active = [None] * len(steps)

            active[0] = pymc.Deterministic(f'Active in {steps[0]}', pymc.MutableData('', prefill[0]))

            for step, step_name in enumerate(steps[:-1]):
                betas[step] = pymc.Beta(f'% {step_name} -> {steps[step + 1]}', priors[step].alpha,
                                        priors[step].beta)

                retained = binomials[step - 1] if step >= 1 else 0
                binomials[step] = pymc.Binomial(f'Number Retained from {step_name} to {steps[step + 1]}', p=betas[step],
                                                n=prefill[step] + retained)

                active[step + 1] = pymc.Deterministic(f'{steps[step + 1]}', prefill[step + 1] + binomials[step])

        return model


    def update_priors(priors : List[BetaPrior], statuses: List[StepStatus]):

        new_priors = [None] * len(priors)

        for i, p in enumerate(priors):
            # number of retains for each step is the sum of all actives and drops in future steps
            retained = sum(statuses[j].active + statuses[j].dropped for j in range(i + 1, len(statuses)))
            alpha = p.alpha + retained
            beta = p.beta + statuses[i].dropped
            new_priors[i] = BetaPrior(**{'alpha':alpha, 'beta':beta })

        return new_priors
