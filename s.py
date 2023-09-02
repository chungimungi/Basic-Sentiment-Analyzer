import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from textblob import TextBlob
import pandas as pd


#OpenAI API key
openai_api_key = "YOUR_OPEN_API_KEY"

llm = OpenAI(openai_api_key=openai_api_key)
agent = create_csv_agent(llm=llm, path="sa.csv", verbose=True)

# Read the customer reviews from the file
df = pd.read_csv("IMDB Dataset.csv.csv")

# Create a Streamlit app
st.title("Customer Reviews Sentiment Analysis")

# Add a query input to the Streamlit app
query = st.text_input("Enter a query:")

if query:
    # Perform sentiment analysis on the query
    
    sentiment = TextBlob(query)
    sentiment_score = sentiment.sentiment.polarity
    sentiment_label = "Positive" if sentiment_score > 0 else "Neutral" if sentiment_score == 0 else "Negative"

    st.write(f"Sentiment: {sentiment_label} (Polarity Score: {sentiment_score:.2f})")

# Add a button to run sentiment analysis on the csv file
button = st.button("Run sentiment analysis on csv file")

if button:
    # Perform sentiment analysis on the csv file
    for i, review in enumerate(df, start=1):
        st.subheader(f"df {i}:")
        st.write(df)
        # Perform sentiment analysis using TextBlob
        sentiment = TextBlob(df["Text"].to_string())
        sentiment_score = sentiment.sentiment.polarity

        # Determine sentiment label based on the polarity score
        sentiment_label = "Positive" if sentiment_score > 0 else "Neutral" if sentiment_score == 0 else "Negative"

        st.write(f"Sentiment: {sentiment_label} (Polarity Score: {sentiment_score:.2f})")
        st.write("---")
