import streamlit as st

# Introduction Text
st.title('A/B Testing Dashboard')
st.write('Your current version has a conversion rate of 5%')
st.write('Please input the number of conversions and non-conversions for your new version.')

# Input from the user
conversions = st.number_input('Enter the number of Conversions:', min_value=0, value=0, format='%d')
non_conversions = st.number_input('Enter the number of Non-Conversions:', min_value=0, value=0, format='%d')

# Button to submit input
if st.button('Submit'):
    st.write('The lift of your new solution is :', None)
    st.write('With a confidence of : ', None)
    # You might perform A/B testing calculations or analysis here
