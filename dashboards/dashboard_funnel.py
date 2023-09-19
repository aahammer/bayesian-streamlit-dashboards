import streamlit as st
import pymc as pm
import numpy as np
import plotly.graph_objects as go

import models.funnel_model as funnel
from models.funnel_model import BetaPrior

st.set_page_config(layout="wide")


def run_model(num_contacts, num_contact_dropouts, num_meetings, num_meeting_dropouts):

    model = funnel.create(['Contacts', 'Pitches', 'Term Sheets'],
                          [BetaPrior(**{'alpha': 3., 'beta': 3.}), BetaPrior(**{'alpha': 2., 'beta': 2.})],
                          prefill=[30, 0, 0]
                          )
    # add pymc model
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

    values = [100, 100 * avg_c_m, (1 - result) * 100]
    labels = ["Pending Contacts", "Ongoing Meetings", "Deal Chance"]
    text_values = [f'{num_contacts} pending <br> Conversion ~{avg_c_m:.1%}'
        , f'{num_meetings} ongoing <br> ~{avg_m_t:.1%} conversion'
        , f'{1 - result:.1%} chance <br> to close a deal']

    hover_texts = text_values

    fig = go.Figure(go.Funnel(
        x=labels,
        y=values,
        text=text_values,
        textposition=['inside' if v > 25 else 'outside' for v in values],
        textinfo="text",  # Display custom text and percentage
        texttemplate='%{text}',
        hovertext=hover_texts,
        hoverinfo="text",
        orientation='v',
        textfont={"size": 20},
        connector={"fillcolor": 'lightsteelblue'},
        marker={"color": ["mediumslateblue", "mediumslateblue",
                          f"{'mediumslateblue' if (1 - result) * 100 > 90 else 'red'}"]}  # Colors for each funnel stage
    ))

    fig.update_layout(
        yaxis={'side': 'right'}
    )

    return 1 - result, fig


def main():
    st.title("Venture Capital Funnel")

    st.text("""
    A company wants to pitch for Venture Capital. The department in charge is highly optimistic on often they will 
    get invited for a pitch and how likely they will end up with a term sheet offer. Management is less optimistic 
    but does not want to get into a fight with the department experts. Instead it is decided to update the beliefs 
    on the go, based on how many contacts get declined and pitches failed. 
    
    The dashboard shows the chance on a deal, given the ongoing and past efforts.
    ⚠️ Less than 90% deal chance will raise a red flag.""")

    cols = st.columns([0.15, 0.05, 0.8, 0.05])

    fig = None

    with cols[0]:
        # Get user input
        num_contacts = st.number_input("Pending Contacts", value=0)
        num_contact_dropouts = st.number_input("Declined Contacts", value=0)
        num_meetings = st.number_input("Ongoing Pitches", value=0)
        num_meeting_dropouts = st.number_input("Failed Pitches", value=0)

        deal_chance = 0

    deal_chance, fig = run_model(num_contacts, num_contact_dropouts, num_meetings, num_meeting_dropouts)

    # Display the image in Streamlit
    config = {
        'staticPlot': True,  # set to True if the chart should be static

    }

    current_height = fig.layout.height or 400
    new_height = 0.8 * current_height

    fig.update_layout(height=new_height, margin=dict(t=20, b=0, l=0, r=0), xaxis=dict(tickfont=dict(size=20)))

    with cols[2]:
        # st.image(image, caption="Static Plotly Chart", use_column_width=True)
        st.plotly_chart(fig, use_container_width=True, config=config)


if __name__ == "__main__":
    main()
