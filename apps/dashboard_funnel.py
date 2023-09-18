import streamlit as st
import pymc as pm
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

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


    avg_closure_potential = prior_potential_closures.mean()
    avg_c_m = prior.prior['% meeting invitation'].values.flatten().mean()
    avg_m_t = prior.prior['% term sheet offer'].values.flatten().mean()
    avg_t_c = prior.prior['% close'].values.flatten().mean()

    values = [100, 100 * avg_c_m, 100 * avg_c_m * avg_m_t, (1-result)*100]
    labels = ["Contacted", "Meetings Ongoing", "Term Sheets Offered", "Deal Chance"]
    text_values = [f'{num_contacts} ongoing <br> ~{avg_c_m:.1%} conversion'
        , f'{num_meetings} ongoing <br> ~{avg_m_t:.1%} conversion'
        , f'{num_term_sheets} ongoing <br> ~{avg_t_c:.1%} conversion'
        , f'{1-result:.1%} chance <br> to close a deal']

    hover_texts = text_values

    fig = go.Figure(go.Funnel(
        x=labels,
        y=values,
        text=text_values,
        textposition=['inside'] * 3 + ['outside'],
        textinfo="text",  # Display custom text and percentage
        texttemplate='%{text}',
        hovertext=hover_texts,
        hoverinfo="text",
        orientation='v'
    ))

    fig.update_layout(
        yaxis={'side': 'right'}
    )

    return 1-result, fig
def main():

    st.title("The chance to close a deal is")

    result_placeholder = st.empty()

    cols = st.columns([0.15, 0.05, 0.8, 0.05])

    fig=None

    with cols[0]:
        # Get user input
        num_contacts = st.number_input("Pending First Contacts", value=0)
        num_contact_dropouts = st.number_input("Rejected after First Contact", value=0)
        num_meetings = st.number_input("Ongoing Meetings", value=0)
        num_meeting_dropouts = st.number_input("Rejected after Meetings", value=0)
        num_term_sheets = st.number_input("Term Sheet Offers", value=0)

        deal_chance = 0
        # Button to execute logic
        if st.button('Recalcualte Chances'):
            deal_chance, fig = run_model(num_contacts, num_contact_dropouts, num_meetings, num_meeting_dropouts, num_term_sheets)

    result_placeholder.write(f'{deal_chance:.2%}')

    # Convert the Plotly figure to an image format (PNG)
    #image_bytes = fig.to_image(format="png")

    # Convert bytes to PIL Image to display in Streamlit
    #image = Image.open(io.BytesIO(image_bytes))

    # Display the image in Streamlit
    config = {
        'staticPlot': True,  # set to True if the chart should be static
        'displayModeBar': True,  # set to False to hide the mode bar (e.g., zoom, save, etc.)
        'displaylogo': False,  # set to False to hide the Plotly logo
        'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d',
                                   'zoomIn2d', 'zoomOut2d', 'autoScale2d',
                                   'resetScale2d', 'toggleSpikelines',
                                   'hoverClosestCartesian', 'hoverCompareCartesian',
                                   'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation',
                                   'handleDrag3d', 'resetCameraDefault3d', 'resetCameraLastSave3d',
                                   'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
                                   'toImage', 'sendDataToCloud', 'toggleHover', 'resetViewMapbox',
                                   'hoverClosest3d', 'hoverClosestGl2d', 'hoverClosestPie',
                                   'toggleHover', 'resetViews', 'toggleHover', 'resetViewSankey']
    }

    current_height = fig.layout.height or 450
    new_height = 1.1 * current_height

    fig.update_layout(height=new_height)

    with cols[2]:
        #st.image(image, caption="Static Plotly Chart", use_column_width=True)
        st.plotly_chart(fig, use_container_width=True, config=config)


if __name__ == "__main__":
    main()
