import streamlit as st
import pymc as pm
import numpy as np

def run_model(num_contacts, num_contact_dropouts, num_meetings, num_meeting_dropouts, num_term_sheets):
    # add pymc model
    rng = np.random.default_rng(43)

    c_m_alpha = 8
    c_m_beta = 29
    m_t_alpha = 26
    m_t_beta = 38
    t_c_alpha = 87
    t_c_beta = 10

    c_m_alpha_update = num_meetings + num_meeting_dropouts + num_term_sheets
    c_m_beta_update = num_contact_dropouts
    m_t_alpha_update = num_term_sheets
    m_t_beta_update = num_meeting_dropouts

    dashboard_model = pm.Model()

    with dashboard_model:
        c_m = pm.Beta('% meeting invitation', alpha=c_m_alpha + c_m_alpha_update, beta=c_m_beta + c_m_beta_update)
        m_t = pm.Beta('% term sheet offer', alpha=m_t_alpha + m_t_alpha_update, beta=m_t_beta + m_t_beta_update)
        t_c = pm.Beta('% close', alpha=t_c_alpha, beta=t_c_beta)

        contact_potential = pm.Binomial('# of meetings', p=c_m, n=num_contacts)
        meeting_potential = pm.Binomial('# of term sheets', p=m_t, n=contact_potential + num_meetings)
        closure_potential = pm.Binomial('# of available deals', p=t_c, n=meeting_potential + num_term_sheets)

        prior = pm.sample_prior_predictive(samples=10_000, random_seed=rng)

    prior_potential_closures = prior.prior['# of available deals'].values.flatten()
    result = len(prior_potential_closures[prior_potential_closures == 0]) / 10_000

    return 1-result
def main():

    # Try to retrieve saved user input from the session state
    if 'num_contacts' not in st.session_state:
        st.session_state.num_contacts = 0  # Default value

    st.title("The chance to close a deal is")

    result_placeholder = st.empty()

    # Get user input
    st.subheader("Enter the number of Venture Capital Firms for each:")
    num_contacts = st.number_input("Pending First Contacts", value=st.session_state.num_contacts)
    num_contact_dropouts = st.number_input("Rejected after First Contact", value=0)
    num_meetings = st.number_input("Ongoing Meetings", value=0)
    num_meeting_dropouts = st.number_input("Rejected after Meetings", value=0)
    num_term_sheets = st.number_input("Term Sheet Offers", value=0)

    result = 0
    # Button to execute logic
    if st.button('Recalcualte Chances'):
        result = run_model(num_contacts, num_contact_dropouts, num_meetings, num_meeting_dropouts, num_term_sheets)

    result_placeholder.write(f'{result:.2%}')

    st.session_state.num_contacts = num_contacts


if __name__ == "__main__":
    main()
