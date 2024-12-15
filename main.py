import streamlit as st
import pandas as pd
from core.bleu_evaluator import calculate_bleu_scores

st.title("BLEU Score Evaluator")
st.write("Evaluate BLEU scores for predictions against reference texts in English or Korean.")
st.write("More metrics will be added onwards.")

# Choose language
lang = st.radio("Select Language", options=["English", "Korean"], horizontal=True)
lang_code = "en" if lang == "English" else "ko"

st.write("CSV must contain 'reference' and 'prediction' columns.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.info("File successfully uploaded!")
    
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")


    st.write("### Input Data")
    st.dataframe(df)

    # Calculate BLEU 
    if "reference" in df.columns and "prediction" in df.columns:
        bleu_scores = calculate_bleu_scores(df, "reference", "prediction", lang=lang_code)
        df["BLEU"] = bleu_scores

        st.write("### Results with BLEU Scores")
        st.dataframe(df)

        # Show average BLEU score
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        st.success(f"Average BLEU Score: {avg_bleu:.4f}")
    else:
        st.error("CSV must contain 'reference' and 'prediction' columns.")
