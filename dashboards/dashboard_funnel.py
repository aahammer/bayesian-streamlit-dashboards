import streamlit as st
import pymc
import numpy
from PIL import Image

import models.funnel_model as funnel
from models.funnel_model import BetaPrior, StepStatus
import widgets.funnel_widget as fw


def main():

    st.title('What''s the chance to get one deal trough the funnel?')

    input_columns = st.columns([0.01,0.79,0.2])

    with input_columns[1]:

        with open("./texts/funnel_case_description.md", "r") as file:
            content = file.read()
            st.markdown(content)


    st.markdown("---")

    input_columns = st.columns(4)

    with input_columns[0]:
        num_contacts = st.number_input("Active Contacts", value=30)
    with input_columns[1]:
        num_contact_dropouts = st.number_input("Dropped Contacts", value=0)
    with input_columns[2]:
        num_meetings = st.number_input("Active Pitches", value=0)
    with input_columns[3]:
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

    fig = fw.create( [num_contacts, num_meetings],  [avg_c_m, avg_m_t], deal_chance)
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown("---")

    st.markdown("[More details on the OKR Analysis Background in Colab](https://colab.research.google.com/drive/16hIZXF7uxrp8gWalPzS_Q7FrL7KIrsqa?usp=sharing)")

    col1, col2 = st.columns([0.1, 4])
    image_github = Image.open('./images/github-mark.png')
    col1.image(image_github, caption='', channels='RGB', output_format='auto', clamp=False, width=24, )
    link = "[Checkout the Code on Github](https://github.com/aahammer/okr-analysis)"
    col2.markdown(link, unsafe_allow_html=True)

    col1, col2 = st.columns([0.1, 4])

    image_linkedin = Image.open('./images/LI-In-Bug.png')
    col1.image(image_linkedin, caption='', channels='RGB', output_format='auto', clamp=False, width=24, )
    link = "[Visit my Linked-In Profile](https://www.linkedin.com/in/andreas-adlichhammer-431a9413a/)"
    col2.markdown(link, unsafe_allow_html=True)


    # image icons would have a nasty expand hover, blocking the actual link.
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)



if __name__ == "__main__":

    st.set_page_config(layout="wide")
    main()