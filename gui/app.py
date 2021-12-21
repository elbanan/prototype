import data
import data_2
import streamlit as st

st.set_page_config(
            page_title=None,
            page_icon=None,
            layout="wide",
            initial_sidebar_state="auto",
            menu_items={
                         'Get Help': 'https://www.extremelycoolapp.com/help',
                         'About': "RadTorch view."
                         }
                         )

MODULES = {
    "Add Dataset": data,
    "View Datasets": data_2
}

st.sidebar.image("logo4.png", use_column_width=True)
# st.sidebar.title('RADTorch Modules')
s = st.sidebar.selectbox(label='', options=list(MODULES.keys()))
data = MODULES[s]
data.app()
