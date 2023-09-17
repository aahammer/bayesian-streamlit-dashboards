import streamlit as st

def calculate_result( num_contacts=0,  num_meetings=0,  num_term_sheets=0):
    return num_contacts * 1/3 * 0.5 * 0.8 + num_meetings * 0.5 * 0.8 + num_term_sheets * 0.8

def main():
    st.title("Number of deals")

    result_placeholder = st.empty()

    # Get user input
    st.subheader("Enter the number of Venture Capital Firms for each:")
    num_contacts = st.number_input("Pending First Contacts", value=0)
    num_contact_dropouts = st.number_input("Rejected after First Contact", value=0)
    num_meetings = st.number_input("Ongoing Meetings", value=0)
    num_meeting_dropouts = st.number_input("Rejected after Meetings", value=0)
    num_term_sheets = st.number_input("Term Sheet Offers", value=0)

    result = calculate_result(num_contacts, num_meetings,  num_term_sheets)

        # Update the placeholder with the result
    result_placeholder.subheader("Result:")
    result_placeholder.write(result)



if __name__ == "__main__":
    main()
