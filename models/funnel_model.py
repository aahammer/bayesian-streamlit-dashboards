from typing import List, Tuple
from pydantic import BaseModel, confloat

import pymc

class BetaPrior(BaseModel):
    alpha: confloat(ge=0)
    beta: confloat(ge=0)


def create(steps: List[str], priors: List[BetaPrior], prefill: List[int] = None) -> pymc.Model:
    if prefill is None:
        prefill = [0] * len(steps)

    assert len(steps) == len(priors) + 1
    assert len(prefill) == len(steps)
    [BetaPrior(**prior) for prior in priors]

    model = pymc.Model()

    with model:

        betas = [None] * len(priors)
        binomials = [None] * len(steps)
        active = [None] * len(steps)

        active[0] = pymc.Deterministic(f'Active in {steps[0]}', pymc.MutableData('', prefill[0]))

        for step, step_name in enumerate(steps[:-1]):
            betas[step] = pymc.Beta(f'Conversion Rate from {step_name} to {steps[step + 1]}', priors[step]['alpha'],
                                    priors[step]['beta'])

            retained = binomials[step - 1] if step >= 1 else 0
            binomials[step] = pymc.Binomial(f'Number Retained from {step_name} to {steps[step + 1]}', p=betas[step],
                                            n=prefill[step] + retained)

            active[step + 1] = pymc.Deterministic(f'Active in {steps[step + 1]}', prefill[step + 1] + binomials[step])

    return model


def update_priors(prior, status):

    new_prior = [None] * len(prior)

    for i, p in enumerate(prior):
        # number of retains for each step is the sum of all actives and drops in future steps
        retained = sum(status[j]['active'] + status[j]['dropped'] for j in range(i + 1, len(status)))
        alpha = p['alpha'] + retained
        beta = p['beta'] + status[i]['dropped']
        new_prior[i] = {'alpha':alpha, 'beta':beta }

    return new_prior


"""
rng = np.random.default_rng(43)

    c_m_alpha = 3
    c_m_beta = 3
    m_t_alpha = 2
    m_t_beta = 2

    c_m_alpha_update = num_meetings + num_meeting_dropouts
    c_m_beta_update = num_contact_dropouts
    m_t_beta_update = num_meeting_dropouts

    dashboard_model = pm.Model()

    with dashboard_model:
        c_m = pm.Beta('% meeting invitation', alpha=c_m_alpha + c_m_alpha_update, beta=c_m_beta + c_m_beta_update)
        m_t = pm.Beta('% term sheet offer', alpha=m_t_alpha, beta=m_t_beta + m_t_beta_update)

        contact_potential = pm.Binomial('# of meetings', p=c_m, n=num_contacts)
        meeting_potential = pm.Binomial('# of term sheets', p=m_t, n=contact_potential + num_meetings)

        prior = pm.sample_prior_predictive(samples=10_000, random_seed=rng)

    prior_potential_closures = prior.prior['# of term sheets'].values.flatten()
    result = len(prior_potential_closures[prior_potential_closures == 0]) / 10_000


    avg_closure_potential = prior_potential_closures.mean()
    avg_c_m = prior.prior['% meeting invitation'].values.flatten().mean()
    avg_m_t = prior.prior['% term sheet offer'].values.flatten().mean()
"""
