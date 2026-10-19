import streamlit as st
import pickle

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Website design
st.title("📰 Fake News Detection System")

st.write(
    "This project uses Machine Learning "
    "to classify news as Real or Fake."
)

st.info("Enter a news article below to check its prediction.")

# Input box
news = st.text_area(
    "Enter News Content",
    placeholder="Type or paste your news here...",
    height=180
)

# Detect button
if st.button("Detect News", type="primary"):

    if news.strip():

        prediction = model.predict([news])[0]

        if prediction == "REAL":
            st.success("✅ Prediction: REAL NEWS")
        else:
            st.error("⚠️ Prediction: FAKE NEWS")

        st.caption(
            "This is an experimental prediction, "
            "not a guarantee that the news is true or false."
        )

    else:
        st.warning("Please enter some news content.")