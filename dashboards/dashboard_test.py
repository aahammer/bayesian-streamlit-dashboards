import streamlit as st
import pymc
import numpy

import models.funnel_model as funnel
from models.funnel_model import BetaPrior, StepStatus
import widgets.funnel_widget as fw


def main():

    num_contacts = st.number_input("Active Contacts", value=0)
    num_contact_dropouts = st.number_input("Dropped Contacts", value=0)
    num_meetings = st.number_input("Active Pitches", value=0)
    num_meeting_dropouts = st.number_input("Dropped Pitches", value=0)

    updated_priors = funnel.update_priors(
        [BetaPrior(**{'alpha': 3., 'beta': 3.}), BetaPrior(**{'alpha': 2., 'beta': 2.})],
        [StepStatus(**{'active': num_contacts, 'dropped': num_contact_dropouts}),
         StepStatus(**{'active': num_meetings, 'dropped': num_meeting_dropouts})]
    )

    model = funnel.create(['Contacts', 'Pitches', 'Term Sheets'],
                          updated_priors,
                          prefill=[num_contacts, num_meetings, 0]
                          )

    rng = numpy.random.default_rng(43)
    with model:
        prior = pymc.sample_prior_predictive(samples=10_000,random_seed=rng)

    prior_potential_closures = prior.prior['Term Sheets'].values.flatten()
    deal_chance= len(prior_potential_closures[prior_potential_closures > 0]) / len(prior_potential_closures)

    avg_closure_potential = prior_potential_closures.mean()
    avg_c_m = prior.prior['% Contacts -> Pitches'].values.flatten().mean()
    avg_m_t = prior.prior['% Pitches -> Term Sheets'].values.flatten().mean()

    #st.text(f'Deal Chance: {deal_chance:.2%}')
    #st.text(f'Avg amount of Deals: {avg_closure_potential}')
    #st.text(f'% Contacts -> Pitches: {avg_c_m}')
    #st.text(f'% Pitches -> Term Sheets {avg_m_t}')

    fig = fw.create( [num_contacts, num_meetings],  [avg_c_m, avg_m_t], deal_chance)
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})



if __name__ == "__main__":

    st.set_page_config(layout="wide")
    main()