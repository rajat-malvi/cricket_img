"""
Login page for the Cricket Image Chatbot
"""

import streamlit as st
from auth import register_user, login_user, initialize_auth_session_state

def show_login_page():
    """
    Display the login page

    Returns:
        True if the user is authenticated, False otherwise
    """
    # Initialize authentication session state
    initialize_auth_session_state()

    # If user is already authenticated, return True
    if st.session_state.is_authenticated:
        return True

    st.title("🏏 Cricket Image Chatbot")
    st.markdown("Please log in or sign up to continue")

    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # Login tab
    with tab1:
        st.subheader("Login")

        # Login form
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if not email or not password:
                    st.error("Please fill in all fields")
                else:
                    success, message, user_data = login_user(email, password)

                    if success:
                        st.session_state.user = user_data
                        st.session_state.is_authenticated = True
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

    # Signup tab
    with tab2:
        st.subheader("Sign Up")

        # Signup form
        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button("Sign Up")

            if submit_button:
                if not name or not email or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(name, email, password)

                    if success:
                        st.success(message)
                        st.info("Please log in with your new account")
                    else:
                        st.error(message)

    return False

if __name__ == "__main__":
    show_login_page()
