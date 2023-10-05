import streamlit as st

import pymc
import numpy
import pandas

import models.a_b_model as a_b
from models.a_b_model import Effect
from models.types import BetaPrior


# Check if 'variants' is already in the session state, if not initialize it
if 'variants' not in st.session_state:
    st.session_state.variants = [{"conversions": 550, "non_conversions": 9450} for _ in range(5)]
if 'control' not in st.session_state:
    st.session_state.control = {"conversions": 5000, "non_conversions": 95000}

# Introduction Text
st.title('Multivariate A/B Testing Dashboard')

st.subheader(f"Control Variant")
col1, col2 = st.columns(2)
st.session_state.control['conversions'] = col1.number_input(
    f'Observed number of conversions',
    min_value=1,
    value=st.session_state.control["conversions"],
    key=f"conversions_control",
    format='%d'
)
st.session_state.control['non_conversions'] = col2.number_input(
    f'Observed number of non-conversions',
    min_value=1,
    value=st.session_state.control["non_conversions"],
    key=f"non_conversions_control",
    format='%d'
)

control_conversion_rate = st.session_state.control['conversions'] / (
            st.session_state.control['conversions'] + st.session_state.control['non_conversions'])
prior_alpha = (control_conversion_rate * 100)
prior_beta = ((1 - control_conversion_rate) * 100)
st.caption(f'* Alpha and Beta Priors for the Variants are set to {prior_alpha:.4} and {prior_beta:.4}' )


st.markdown("---")

selected_variants = st.slider("Number of variants", min_value=1, max_value=5, value=1, step=1)


col1, col2, _= st.columns([0.25,0.25,0.5])

# Display number inputs for each variant
for i in range(selected_variants):
    st.subheader(f"Variant {i + 1}")
    col1, col2 = st.columns(2)
    st.session_state.variants[i]["conversions"] = col1.number_input(
        f'Observed number of conversions',
        min_value=1,
        value=st.session_state.variants[i]["conversions"],
        key=f"conversions_{i}",
        format='%d'
    )
    st.session_state.variants[i]["non_conversions"] = col2.number_input(
        f'Observed number of non-conversions',
        min_value=1,
        value=st.session_state.variants[i]["non_conversions"],
        key=f"non_conversions_{i}",
        format='%d'
    )

st.markdown("---")

# Button to submit input
if st.button('Evaluate Variants', type="primary"):

    variant_priors = []

    for i in range(selected_variants):
        variant_priors.append(BetaPrior(**{'alpha': st.session_state.variants[i]['conversions'] + prior_alpha, 'beta': st.session_state.variants[i]['non_conversions'] + prior_beta}))

    model = a_b.create( [BetaPrior(**{'alpha':st.session_state.control['conversions'], 'beta':st.session_state.control['non_conversions']})] + variant_priors)
    with model:
        prior = pymc.sample_prior_predictive(samples=10_000, random_seed=43)

    # retrieve the variant samples from the prior trace and calcualte the effects
    variants = prior['prior']['p'].values.reshape(-1, selected_variants+1).T[1:]
    effects = a_b.evaluate_variants(prior['prior']['p'].sel(p_dim_0=0).values.flatten(), variants)


    st.header("Variant Evaluation:")
    # set index names to variant names
    df = pandas.DataFrame([effect.dict() for effect in effects])
    df.index = ['Variant ' + str(i + 1) for i in range(len(df))]

    # streamlit does not support pandas format settings
    df['lift'] = (df['lift'] * 100).apply('{:.2f}%'.format)
    df['confidence'] = (df['confidence'] * 100).apply('{:.2f}%'.format)
    df.columns = ['Mean Lift Over Control Variant', 'Confidence in Outperforming Control Variant']


    st.table(df)

    st.caption('* small deviations between equal inputs are normal, because of the not 100% deterministic Monte-Carlo Sampling')
