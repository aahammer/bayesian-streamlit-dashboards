import streamlit as st

import pymc
import numpy

import models.a_b_model as a_b
from models.a_b_model import Effect
from models.types import BetaPrior


# Introduction Text
st.title('A/B Testing Dashboard')
st.write('Your current version has a conversion rate of 5% (strong prior)')
st.write('The initial prior for the new versions is also 5% (weak prior)')
st.write('Please input the number of conversions and non-conversions for your new version.')

# Input from the user
#conversions = st.number_input('Enter the number of Conversions:', min_value=1, value=1, format='%d')
#non_conversions = st.number_input('Enter the number of Non-Conversions:', min_value=1, value=9, format='%d')


st.markdown("---")

# Check if 'variants' is already in the session state, if not initialize it
if 'variants' not in st.session_state:
    st.session_state.variants = [{"conversions": 1, "non_conversions": 9}]

col1, col2, _= st.columns([0.25,0.25,0.5])
# Button to add a variant
if col1.button("Add variant"):
    st.session_state.variants.append({"conversions": 1, "non_conversions": 9})

# Button to remove a variant
if col2.button("Remove variant") and len(st.session_state.variants) > 1:
    st.session_state.variants.pop()

# Display number inputs for each variant
for i, variant in enumerate(st.session_state.variants):
    st.subheader(f"Variant {i + 1}")
    st.session_state.variants[i]["conversions"] = st.number_input(
        f'Enter the number of Conversions for variant {i + 1}:',
        min_value=1,
        value=variant["conversions"],
        key=f"conversions_{i}",
        format='%d'
    )
    st.session_state.variants[i]["non_conversions"] = st.number_input(
        f'Enter the number of Non-Conversions for variant {i + 1}:',
        min_value=1,
        value=variant["non_conversions"],
        key=f"non_conversions_{i}",
        format='%d'
    )

# Button to submit input
if st.button('Submit'):

    variant_priors = []

    for v in st.session_state.variants:
        variant_priors.append(BetaPrior(**{'alpha': v['conversions'] + 1, 'beta': v['non_conversions'] + 20}))

    model = a_b.create( [BetaPrior(**{'alpha':500, 'beta':9_500})] + variant_priors)
    with model:
        prior = pymc.sample_prior_predictive(samples=10_000, random_seed=43)

    # retrieve the variant samples from the prior trace and calcualte the effects
    variants = prior['prior']['p'].values.reshape(-1, len(st.session_state.variants)+1).T[1:]
    effects = a_b.evaluate_variants(prior['prior']['p'].sel(p_dim_0=0).values.flatten(), variants)

    for e in effects:
        st.write('The lift of your new solution is about :', f'{e.lift:.2%}')
        st.write('With a confidence to be better of : ', f'{e.confidence:.2%}')





'''
import pandas as pd
# Example data: a list of 3 columns
    data = [
        ['A1', 'B1', 'C1'],
        ['A2', 'B2', 'C2'],
        ['A3', 'B3', 'C3'],
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Column A', 'Column B', 'Column C'])

    # Display in Streamlit
    st.table(df)
'''