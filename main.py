import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not available
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment using VADER."""
    scores = sia.polarity_scores(text)
    return scores

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV file and analyze sentiment."""
    df = pd.read_csv(uploaded_file)
    
    # Determine which column to use for sentiment analysis
    valid_columns = ['Feedback', 'Review', 'Textgi']
    selected_column = next((col for col in valid_columns if col in df.columns), None)
    
    if not selected_column:
        st.error("CSV must contain a 'Feedback', 'Review', or 'Text' column.")
        return None
    
    # Apply sentiment analysis
    df[['positive', 'negative', 'neutral', 'compound']] = df[selected_column].apply(lambda text: pd.Series(analyze_sentiment(str(text))))
    return df

def plot_sentiment_charts(df):
    """Plot sentiment distribution as both a pie chart and a bar chart."""
    sentiment_counts = {
        'Positive': (df['positive'] > df[['negative', 'neutral']].max(axis=1)).sum(),
        'Negative': (df['negative'] > df[['positive', 'neutral']].max(axis=1)).sum(),
        'Neutral': (df['neutral'] > df[['positive', 'negative']].max(axis=1)).sum()
    }
    
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#3bed68', '#ed184d', '#1776eb']
    
    # Create layout for two charts
    col1, col2 = st.columns(2)
    
    # Pie Chart
    with col1:
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title('Sentiment Distribution')
        st.pyplot(plt)
    
    # Bar Chart
    with col2:
        plt.figure(figsize=(4.5, 5))
        sns.barplot(x=labels, y=sizes, palette=colors)
        plt.ylabel("Count")
        plt.title("Sentiment Distribution")
        st.pyplot(plt)
        
    # Summary Text
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    st.write(f"### Summary: The overall sentiment of the data is **{dominant_sentiment}**.")

# Streamlit UI
st.title("Sentiment Analysis Web Application")

# Tabs for different functionalities
tabs = st.tabs(["Upload CSV", "Text Input"])

# Part 1: File Upload Sentiment Analysis
with tabs[0]:
    st.header("Upload CSV for Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Feedback', 'Review', or 'Text' column", type=["csv"])
    
    if uploaded_file:
        df_result = process_uploaded_file(uploaded_file)
        if df_result is not None:
            st.write("### Sentiment Analysis Results")
            st.dataframe(df_result)
            plot_sentiment_charts(df_result)

# Part 2: Text Input Sentiment Analysis
with tabs[1]:
    st.header("Type Your Text for Sentiment Analysis")
    user_text = st.text_area("Enter text here and press enter:")
    
    if st.button("Calculate", help="Click to analyze sentiment"):
        if user_text.strip():  # Ensure input is not empty
            result = analyze_sentiment(user_text)
            st.write("### Sentiment Scores")
            
            def sentiment_bar(label, value, color):
                """Function to create a sentiment bar visualization"""
                bar_html = f"""
                <div style="margin-bottom: 1rem">
                    <div style="width: {value*100}%; height: 10px; background-color: {color}; padding: 2px; border-radius: 5px; text-align: center; color: white;">  
                    </div>
                    {label}: {value:.1f}
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)
            
            # Displaying sentiment bars
            sentiment_bar("Negative", result['neg'], "#ed187b")
            sentiment_bar("Neutral", result['neu'], "#1776eb")
            sentiment_bar("Positive", result['pos'], "#3bed68")
        else:
            st.warning("Please enter some text before calculating.")


