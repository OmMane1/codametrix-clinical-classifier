"""Interactive explainable-classifier demo (the presentation moment).

Run:  pip install streamlit
      streamlit run app.py

Paste a clinical note -> see the predicted specialty, the confidence over all
five classes, and the note with the words that drove the decision highlighted.
"""
import joblib
import streamlit as st

from clinical_classifier.explain import global_top_terms, render_html, token_contributions

MODEL_PATH = "models/tfidf_submission.joblib"

st.set_page_config(page_title="Clinical Note Classifier", layout="centered")
st.title("🩺 Clinical Note Classifier — explainable")
st.caption("TF-IDF + Logistic Regression. Every prediction is fully explainable: "
           "the score is an exact sum of per-word contributions.")


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


model = load()

text = st.text_area("Paste a clinical note:", height=220,
                    placeholder="ADMISSION DIAGNOSES, 1. ...")

if st.button("Classify", type="primary") and text.strip():
    pred, proba, _ = token_contributions(model, text)
    conf = max(proba.values())

    st.subheader(f"Prediction: {pred}")
    if conf < 0.60:
        st.warning(f"Low confidence ({conf:.0%}) — in a real deployment this note "
                   "would be flagged for human review.")
    st.bar_chart({"probability": proba})
    st.markdown("**Why?** Words driving the prediction:")
    st.markdown(render_html(model, text), unsafe_allow_html=True)

with st.expander("What the model learned each specialty looks like"):
    for cls, terms in global_top_terms(model, n=12).items():
        st.markdown(f"**{cls}** — " + ", ".join(t for t, _ in terms))
