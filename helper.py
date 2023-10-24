import wordcloud
from wordcloud import WordCloud
from sklearn.metrics import mean_squared_error, r2_score

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages

    num_messages = df.shape[0]
    words = []

    # this will fetch the total no of messages
    for messages in df['messages']:
        words.extend(messages.split())

    # this is going to fetch no of media messages
    num_media_messages = df[df['messages'] == '<Media omitted>\n'].shape[0]

    # this is going to fetch no of emoji

    # Define a function to count emojis in a text
    def count_emojis(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return len(re.findall(emoji_pattern, text))

    # Count the total emojis in the "messages" column
    total_emojis = df['messages'].apply(count_emojis).sum()

    return num_messages, len(words), num_media_messages, total_emojis


# code for displaying most active user in stats form and graph form
def most_busy_users(df):
    x = df['user'].value_counts().head(10)
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'Name', 'count': 'Percent'})
    return x, df


# code for displaying most used words by the help of word cloud
def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width=300, height=300, min_font_size=10, background_color="white")
    df_wc = wc.generate(df['messages'].str.cat(sep=" "))

    return df_wc


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['messages'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['messages'].reset_index()

    return daily_timeline


def perform_linear_regression(df):
    X = df[['hour']]
    y = df['message_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pre = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pre)

    return model, mae


def calculate_regression_metrics(model, df):
    y_pred = model.predict(df[['hour']])
    rmse = np.sqrt(mean_squared_error(df['message_count'], y_pred))
    r_squared = r2_score(df['message_count'], y_pred)
    return rmse, r_squared