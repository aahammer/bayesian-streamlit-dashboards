import streamlit as st

import pymc
import numpy

import models.a_b_model as a_b
from models.a_b_model import Effect
from models.types import BetaPrior


# Introduction Text
st.title('A/B Testing Dashboard')
st.write('Your current version has a conversion rate of 5%')
st.write('Please input the number of conversions and non-conversions for your new version.')

# Input from the user
conversions = st.number_input('Enter the number of Conversions:', min_value=1, value=1, format='%d')
non_conversions = st.number_input('Enter the number of Non-Conversions:', min_value=1, value=9, format='%d')


# Button to submit input
if st.button('Submit'):

    model = a_b.create( [BetaPrior(**{'alpha':500, 'beta':9_500}), BetaPrior(**{'alpha':conversions, 'beta':non_conversions})])
    with model:
        prior = pymc.sample_prior_predictive(samples=10_000, random_seed=43)

    effects = a_b.evaluate_variants(prior['prior']['p'].sel(p_dim_0=0).values.flatten(), [prior['prior']['p'].sel(p_dim_0=1).values.flatten()])

    st.write('The lift of your new solution is about :', f'{effects[0].lift:.2%}')
    st.write('With a confidence to be better of : ', f'{effects[0].confidence:.2%}')
    # You might perform A/B testing calculations or analysis here
