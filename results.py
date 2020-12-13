from collections import Counter

import pandas as pd
import plotly.express as px
from datacleaner import DataCleaner


class Results(object):

    def __init__(self):
        self.classes = ['Negative', 'Neutral', 'Positive']
        self.tweets_num = 0
        self.positive_tweets_num = 0
        self.negative_tweets_num = 0
        self.neutral_tweets_num = 0
        self.dataclean = DataCleaner()

    def calculate_results(self, df):

        dataframe_class, dataframe_tweets = df['sentiment_class'], df['tweet']
        dataframe_value_count = dataframe_class.value_counts()

        dictionary = dict(zip(dataframe_value_count.index, dataframe_value_count.values))

        for key, value in dictionary.items():
            if key == "Negative":
                self.negative_tweets_num = value
            elif key == "Positive":
                self.positive_tweets_num = value
            else:
                self.neutral_tweets_num = value

        self.tweets_num = dataframe_class.count()
        self.dataframe_tweets = self.dataclean.prepare_data_set_without_stem(list(dataframe_tweets))
        self.most_words = Counter(self.dataframe_tweets).most_common(15)

        self.dataframe_tweets_negative = self.dataclean.prepare_data_set_without_stem(
            list(df[df['sentiment_class'] == 'Negative']['tweet']))
        self.dataframe_tweets_positive = self.dataclean.prepare_data_set_without_stem(
            list(df[df['sentiment_class'] == 'Positive']['tweet']))

        df['date'] = pd.to_datetime(df.created_at).apply(lambda x: x.date())
        sub = df[['sentiment_class', 'date', 'tweet']]
        self.date_wise_counts = sub.groupby(['date', 'sentiment_class']) \
            .count() \
            .reset_index()

    def get_bar_chart_counts(self):
        data = {'Sentiment': self.classes,
                'Number of tweets': [self.negative_tweets_num, self.neutral_tweets_num, self.positive_tweets_num]}


        fig = px.bar(data, x='Sentiment', y='Number of tweets', title='Distribution of Tweet Sentiment')

        return fig

    def get_pie_chart_counts(self):
        data = {'Sentiment': self.classes,
                'Number of tweets': [self.negative_tweets_num, self.neutral_tweets_num, self.positive_tweets_num]}

        fig = px.pie(data, values='Number of tweets', names='Sentiment', title='Distribution of Tweet Sentiment')
        return fig

    def get_bar_chart_most_counts(self):
        data = pd.DataFrame(self.most_words, columns=['Words', "Frequency"])

        fig = px.bar(data, x='Words', y='Frequency', title='Most occuring words in tweets')
        return fig

    def get_pie_chart_most_counts(self):
        data = pd.DataFrame(self.most_words, columns=['Words', "Frequency"])

        fig = px.pie(data, values='Frequency', names='Words', title='Most occuring words in tweets')
        return fig

    def get_line_chart_tweets(self):

        fig = px.line(self.date_wise_counts, x="date", y="tweet", color='sentiment_class')
        return fig

        pass

    def get_stats_table(self):

        df = pd.DataFrame({'Sentiment': ['Total Tweets'],
                           'Number of tweets': [self.negative_tweets_num + self.neutral_tweets_num +
                                                self.positive_tweets_num]})
        df = df.append(pd.DataFrame({'Sentiment': self.classes,
                                     'Number of tweets': [self.negative_tweets_num, self.neutral_tweets_num,
                                                          self.positive_tweets_num]}), ignore_index=True)

        return df

    def get_stats_table_most_counts(self):

        df = pd.DataFrame(self.most_words, columns=['Words', "Frequency"])

        return df

    def get_word_cloud(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self.dataframe_tweets))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=50, background_color="white").generate(
            artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def get_word_cloud_negative(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self.dataframe_tweets_negative))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=30, background_color="white").generate(
            artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def get_word_cloud_positive(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self.dataframe_tweets_positive))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=30, background_color="white").generate(
            artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
