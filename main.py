import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

# Check if CUDA is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1

# Initialize sentiment analysis pipeline with RoBERTa
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=True,  # Return scores for all sentiment classes
        device=device  # Use GPU if available
    )

sentiment_pipeline = load_sentiment_model()

def analyze_sentiment(text):
    """Analyze sentiment using RoBERTa."""
    # Handle long texts by splitting into chunks if necessary
    max_length = 512  # RoBERTa's max token length
    # Approximate character count based on tokens
    if len(text) > max_length * 4:  
        chunks = [text[i:i+max_length*4] for i in range(0, len(text), max_length*4)]
        results = []
        for chunk in chunks:
            results.append(sentiment_pipeline(chunk)[0])
        
        # Average the scores across chunks
        neg = sum(r[0]['score'] for r in results) / len(results)
        neu = sum(r[1]['score'] for r in results) / len(results)
        pos = sum(r[2]['score'] for r in results) / len(results)
    else:
        result = sentiment_pipeline(text)[0]
        neg = result[0]['score']  # Negative score (label: 'negative')
        neu = result[1]['score']  # Neutral score  (label: 'neutral')
        pos = result[2]['score']  # Positive score (label: 'positive')
    
    # Calculate compound as normalized difference (similar to VADER)
    compound = pos - neg
    
    return {
        'pos': pos,
        'neg': neg, 
        'neu': neu,
        'compound': compound
    }

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV file and analyze sentiment."""
    df = pd.read_csv(uploaded_file)
    
    # Determine which column to use for sentiment analysis
    valid_columns = ['Feedback', 'Review', 'Text']
    selected_column = next((col for col in valid_columns if col in df.columns), None)
    
    if not selected_column:
        st.error("CSV must contain a 'Feedback', 'Review', or 'Text' column.")
        return None
    
    # Create a progress bar
    progress_bar = st.progress(0)
    total_rows = len(df)
    
    # Apply sentiment analysis with progress updates
    results = []
    for i, text in enumerate(df[selected_column]):
        scores = analyze_sentiment(str(text))
        results.append(scores)
        # Update progress bar
        progress_bar.progress((i + 1) / total_rows)
    
    # Convert results to DataFrame and join with original df
    scores_df = pd.DataFrame(results)
    df[['positive', 'negative', 'neutral', 'compound']] = scores_df[['pos', 'neg', 'neu', 'compound']]
    
    # Categorize sentiment
    df['Sentiment'] = df.apply(lambda row: "Positive" if row['compound'] > 0.05 
                              else "Negative" if row['compound'] < -0.05 else "Neutral", axis=1)
    
    # Remove progress bar when done
    progress_bar.empty()
    
    return df

def plot_sentiment_charts(df):
    """Plot sentiment distribution as both a pie chart and a bar chart."""
    sentiment_counts = df['Sentiment'].value_counts()
    
    labels = sentiment_counts.index.tolist()
    sizes = sentiment_counts.values.tolist()
    colors = {'positive': '#3bed68', 'negative': '#ed184d', 'neutral': '#1776eb'}
    
    # Convert labels to lowercase to match the keys in the colors dictionary
    color_list = [colors[label.lower()] for label in labels]
    
    # Create layout for two charts
    col1, col2 = st.columns(2)
    
    # Pie Chart
    with col1:
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=color_list, startangle=140)
        plt.title('Sentiment Distribution')
        st.pyplot(plt)
    
    # Bar Chart
    with col2:
        plt.figure(figsize=(4.5, 5))
        sns.barplot(x=labels, y=sizes, palette=color_list)
        plt.ylabel("Count")
        plt.title("Sentiment Distribution")
        st.pyplot(plt)
        
    # Summary Text
    dominant_sentiment = sentiment_counts.idxmax()
    st.write(f"### Summary: The overall sentiment of the data is **{dominant_sentiment}**.")

# Streamlit UI
st.title("Advanced Sentiment Analysis with RoBERTa")
st.markdown("""
This app uses the **cardiffnlp/twitter-roberta-base-sentiment** model, 
which performs better on longer texts than VADER.
""")

# Tabs for different functionalities
tabs = st.tabs(["Upload CSV", "Text Input"])

# Part 1: File Upload Sentiment Analysis
with tabs[0]:
    st.header("Upload CSV for Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Feedback', 'Review', or 'Text' column", type=["csv"])
    
    if uploaded_file:
        with st.spinner("Analyzing sentiment... This may take a moment for large files."):
            df_result = process_uploaded_file(uploaded_file)
            
        if df_result is not None:
            st.write("### Sentiment Analysis Results")
            
            # Display tables for each sentiment category
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                # Determine which column to display
                display_column = next((col for col in ['Review', 'Feedback', 'Text'] if col in df_result.columns), None)
                
                if display_column:
                    filtered_df = df_result[df_result['Sentiment'] == sentiment][[display_column, 'Sentiment', 'positive', 'negative', 'neutral']]
                    if not filtered_df.empty:
                        st.write(f"### {sentiment} Reviews ({len(filtered_df)} total)")
                        st.dataframe(filtered_df.reset_index(drop=True))
            
            plot_sentiment_charts(df_result)
            
            # Option to download the analyzed data
            csv = df_result.to_csv(index=False)
            st.download_button(
                label="Download analyzed data as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )

# Part 2: Text Input Sentiment Analysis
with tabs[1]:
    st.header("Type Your Text for Sentiment Analysis")
    user_text = st.text_area("Enter text here (works well with longer reviews):", height=200)
    
    if st.button("Calculate", help="Click to analyze sentiment"):
        if user_text.strip():  # Ensure input is not empty
            with st.spinner("Analyzing sentiment..."):
                result = analyze_sentiment(user_text)
            
            st.write("### Sentiment Scores")
            
            def sentiment_bar(label, value, color):
                """Function to create a sentiment bar visualization"""
                bar_html = f"""
                <div style="margin-bottom: 1rem">
                    <div style="width: {value*100}%; height: 10px; background-color: {color}; padding: 2px; border-radius: 5px; text-align: center; color: white;">  
                    </div>
                    {label}: {value:.3f}
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)
            
            # Displaying sentiment bars
            sentiment_bar("Negative", result['neg'], "#ed187b")
            sentiment_bar("Neutral", result['neu'], "#1776eb")
            sentiment_bar("Positive", result['pos'], "#3bed68")
            
            # Displaying compound score
            compound_score = result['compound']
            sentiment_label = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"
            st.write(f"Compound Score: {compound_score:.4f}")
            st.write(f"### Overall Sentiment: **{sentiment_label}**")
        else:
            st.warning("Please enter some text before calculating.")