import base64

from tensorflow.python.keras import backend as tb

import pandas as pd
import streamlit as st



from model_inference import Inference
from tweetmanger import TweetManager
from results import Results


@st.cache(allow_output_mutation=True)
def load_model():
    """
     Load classification model and cache it
    """
    with st.spinner('Loading classification model...'):
        classifier = Inference()

    return classifier


@st.cache(allow_output_mutation=True)
def init_twitter():
    """
    Load twitter API endpoint and cache it
    """
    with st.spinner('Loading Twitter Manager...'):
        tweet_manager = TweetManager()

    return tweet_manager


@st.cache(allow_output_mutation=True)
def get_twitter_data(tweet_manager, tweet_input, sidebar_result_type, sidebar_tweet_count):
    """
    Get data from twitter endpoint and cache
    """
    df = tweet_manager.get_tweets(tweet_input, result_type=sidebar_result_type, count=sidebar_tweet_count)

    return df


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download as csv file (Right click and save link as csv)</a>'
    return href


def main():
    """
    Main function called that draws the UI and flow
    """

    # The title definition
    st.title("Sentiment Analyzer on Twitter Hashtags in Arabic Language")

    # The sidebar definitions for configuring the application
    st.sidebar.header("Model options")

    # The type of mode selectbox
    sidebar_model_type = st.sidebar.selectbox('Model Type', ('LSTM', 'SVM','RNN','MLP','LSTM bidirectionnelle'), index=0)

    # Tweet parser options.
    st.sidebar.header("Tweet Parsing options")

    # Slider for configuring the number of tweets to be parsed.
    sidebar_tweet_count = st.sidebar.slider(label='Number of tweets',
                                            min_value=50,
                                            max_value=5000,
                                            value=50,
                                            step=50)

    # The type of tweets to be parsed selectbox, not to be modified.
    sidebar_result_type = st.sidebar.selectbox('Result Type', ('popular', 'mixed', 'recent'), index=0)

    pd.set_option('display.max_colwidth', 0)

    # The models are loaded and kept it memory for optimised performance.
    classifier = load_model()

    # Twitter authentication is done and kept in memory for subsequent API calls
    tweet_manager = init_twitter()

    # Class for drawing the results
    results = Results()

    # Input box for taking in hashtag
    st.subheader('Input the hashtag to analyze')
    tweet_input = st.text_input('Hashtag:')

    if tweet_input != '':
        # Get tweets
        with st.spinner('Parsing from twitter API'):
            # Twitter endpoint is called
            df = get_twitter_data(tweet_manager, tweet_input, sidebar_result_type, sidebar_tweet_count)

        # st.dataframe(df)
        # Make predictions
        if df.__len__() > 0:
            with st.spinner('Predicting...'):
                # If tweets are present, the prediction is done on the dataframe.
                pred = classifier.get_sentiment(df, sidebar_model_type)

                # Predictions and dataframe displayed.
                st.subheader('Prediction:')
                st.dataframe(pred)

                # Download link for the csv file containing predictions.
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            # All the results are calculated
            results.calculate_results(df)

            # ALl visualizations put into the UI
            with st.spinner('Generating Visualizations...'):
                st.header('Visualizations')
                st.subheader('Pie Chart')
                st.plotly_chart(results.get_pie_chart_counts())
                st.subheader('Bar Chart')
                st.plotly_chart(results.get_bar_chart_counts())
                st.subheader('Pie Chart showing most used words')
                st.plotly_chart(results.get_pie_chart_most_counts())
                st.subheader('Bar Chart showing most used words')
                st.plotly_chart(results.get_bar_chart_most_counts())

                st.subheader('Time series showing tweets')
                st.plotly_chart(results.get_line_chart_tweets())

                st.header('Tables')
                st.subheader('Hashtag Analysis')
                st.write('Number of tweets per classification')
                st.table(results.get_stats_table())

                st.subheader('Most words frequency')
                st.write('Top 15 words from tweets')
                st.table(results.get_stats_table_most_counts())

                st.subheader('Word cloud')
                st.write('Top 50 words in all the tweets represented as word cloud')
                results.get_word_cloud()
                st.pyplot()

                st.write('Top 30 words in positive tweets')
                results.get_word_cloud_positive()
                st.pyplot()

                st.write('Top 30 words in negative tweets')
                results.get_word_cloud_negative()
                st.pyplot()


        else:
            # If no tweets found
            st.write('No tweets found')


if __name__ == '__main__':
    main()
