import streamlit as st
from ui.tab.track_speed_tab import track_tab
from ui.tab.incheon_check_tab import check_tab

st.set_page_config(page_title="POC", layout="wide")
st.title("인천공항 POC")
st.write("info")

tab1, tab2 = st.tabs(["check", "track_roi_tab" ])

with tab1:
    check_tab()
# with tab2:
    # track_tab()
    