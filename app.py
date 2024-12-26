import streamlit as st

pg = st.navigation([st.Page("diabetes_app.py"), st.Page("heart_app.py"), st.Page("healthcare_app.py")])
pg.run()
# import streamlit as st

# st.write("Hello ,let's learn how to build a streamlit app together")

# if __name__ == "__main__":
#     main()