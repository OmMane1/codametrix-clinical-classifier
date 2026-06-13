"""Interactive explainable-classifier demo (the presentation moment).

Run:  pip install streamlit
      streamlit run app.py

Pick a sample note (or paste your own) -> see the routing decision, confidence
over all five specialties, the words that drove it, and the note highlighted.
"""
import joblib
import pandas as pd
import streamlit as st

from clinical_classifier.explain import global_top_terms, highlight_html

MODEL_PATH = "models/tfidf_submission.joblib"

st.set_page_config(page_title="Clinical Note Classifier", layout="centered")
st.title("🩺 Clinical Note Classifier — explainable")
st.caption("TF-IDF + Logistic Regression. The prediction is an exact sum of "
           "per-word contributions — no black box, no SHAP approximation.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_examples():
    """One illustrative note per specialty from the held-out val set."""
    val = pd.read_csv("data/splits/val.csv")
    ex = {"— paste your own —": ""}
    for lab in ["Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "Other"]:
        sub = val[val.label == lab]
        if not sub.empty:
            ex[f"Example: {lab}"] = sub.iloc[0]["text"][:1500]
    return ex


model = load_model()
examples = load_examples()

threshold = st.sidebar.slider("Auto-route confidence threshold", 0.3, 0.9, 0.60, 0.05,
                              help="Below this, the note is flagged for a human coder.")

choice = st.selectbox("Try a sample note, or paste your own below:", list(examples))
text = st.text_area("Clinical note:", value=examples[choice], height=200)

if st.button("Classify", type="primary") and text.strip():
    pred, proba, contrib, body = highlight_html(model, text)
    conf = max(proba.values())

    # The deployment decision — the human-in-the-loop story.
    if conf >= threshold:
        st.success(f"✅ **Auto-route → {pred}**  ({conf:.0%} confidence)")
    else:
        st.warning(f"⚠️ **Flag for human review** — top guess {pred}, only {conf:.0%} "
                   f"confidence (below {threshold:.0%}).")

    st.markdown("**Confidence across specialties**")
    st.bar_chart(pd.Series(proba, name="probability"))

    pos = sorted(contrib.items(), key=lambda kv: -kv[1])[:6]
    neg = sorted(contrib.items(), key=lambda kv: kv[1])[:4]
    c1, c2 = st.columns(2)
    c1.markdown("**Supports this call** 🟢\n\n" +
                "\n".join(f"- {w} (`{v:+.2f}`)" for w, v in pos if v > 0))
    c2.markdown("**Argues against** 🔴\n\n" +
                "\n".join(f"- {w} (`{v:+.2f}`)" for w, v in neg if v < 0))

    st.markdown("**The note, with drivers highlighted**")
    st.markdown(body + '<p style="color:#999;font-size:12px">Green = supports, '
                'red = argues against. Hover for the exact weight.</p>',
                unsafe_allow_html=True)

with st.expander("What the model learned each specialty looks like"):
    for cls, terms in global_top_terms(model, n=12).items():
        st.markdown(f"**{cls}** — " + ", ".join(t for t, _ in terms))
