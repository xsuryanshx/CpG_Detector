import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inference_snippet import Inference, CpGPredictor

st.set_page_config(page_title="GC Predictor", page_icon="🔎", layout="centered")

st.title("GC Detector Model")
st.markdown(
    """**Objective** : The project requires you to build a neural network to count the number of CpGs (consecutive CGs) in given DNA (of N, A, C, G, T) sequences. 
    """
)
st.info(
    "For example, for an input string like “NCACANNTNCGGAGGCGNA”, your model should output something like 1.96 or 2.04) "
)
sequence = st.text_input("Enter Sequence", "")
valid_seq = True
for ele in set(sequence):
    if ele not in ["N", "A", "C", "G", "T"]:
        valid_seq = False

if st.button("Predict"):
    if not valid_seq:
        st.error("**Please enter a valid DNA sequence**")
    else:
        predicted_count = Inference().evaluation_for_single_sequence(dna_seq=sequence)
        st.write("The predicted count is: ", np.round(predicted_count, 5))
