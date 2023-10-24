import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import seaborn as sns


def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_messages': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_messages']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['messages'] = messages
    df.drop(columns=['user_messages'], inplace=True)
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['only_date'] = df['date'].dt.day

    # Create a new DataFrame to count messages per hour
    message_by_hour = df.groupby(df['date'].dt.hour).size().reset_index(name='message_count')

    # Merge the new DataFrame with your original DataFrame
    df = df.merge(message_by_hour, left_on=df['date'].dt.hour,
                  right_on=message_by_hour['date'], how='left')

    # Filter the DataFrame to exclude the row with the specified name
    df = df[df['user'] != 'You joined a group via invite in the community']

    # Reset the index of the filtered DataFrame
    df.reset_index(drop=True, inplace=True)

    if 'group_notification' in df['user'].values:
        df = df[df['user'] != 'group_notification']
    #     df.reset_index(drop=True, inplace=True)
    # else:
    #     pass



    # Function for linear regression
    def perform_linear_regression(df):
        X = df[['hour']]
        y = df['message_count']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        return model, mae

    return df
