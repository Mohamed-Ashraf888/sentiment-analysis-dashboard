import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.predict import predict_sentiment
from src.database import insert_prediction, load_predictions, init_db

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
init_db()

st.title("Customer Review Sentiment Analysis Dashboard")
st.write("Enter a customer review and get an instant sentiment prediction.")

review = st.text_area("Write a review", height=140, placeholder="Example: This product is very useful and easy to use.")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            sentiment, confidence = predict_sentiment(review)
            insert_prediction(review, sentiment, confidence)
            st.success(f"Predicted Sentiment: {sentiment}")
            st.info(f"Confidence Score: {confidence:.2f}")
        except Exception as e:
            st.error(str(e))

rows, columns = load_predictions()
df = pd.DataFrame(rows, columns=columns)

st.divider()
st.header("Saved Predictions")

if df.empty:
    st.write("No predictions yet.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        counts = df["predicted_sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Sentiment Percentages")
        percentages = df["predicted_sentiment"].value_counts(normalize=True) * 100
        st.dataframe(percentages.rename("percentage").round(2))

    st.subheader("Word Cloud")
    all_text = " ".join(df["review_text"].astype(str).tolist())
    if all_text.strip():
        wc = WordCloud(width=900, height=350, background_color="white").generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Prediction History")
    st.dataframe(df)
