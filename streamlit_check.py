import streamlit as st

# Set the page title
st.title("My First Streamlit App")

# Add a simple subheader
st.subheader("Welcome to this basic app!")

# Add a text input box
user_name = st.text_input("Enter your name:")

# Add a slider widget
user_age = st.slider("Select your age:", 1, 100, 25)

# Add a button that displays a message when clicked
if st.button("Submit"):
    if user_name:
        st.success(f"Hello {user_name}! You are {user_age} years old.")
    else:
        st.warning("Please enter a name first!")
