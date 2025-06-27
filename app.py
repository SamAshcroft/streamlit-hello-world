import streamlit as st
import numpy as np

dataframe = np.random.randn(10, 20)

st.write("Hey here is my random table WOOP! Hello there")
st.dataframe(dataframe)
st.write("This working?")

st.write("Ok let's see if this works!")