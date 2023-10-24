import streamlit as st
import pre_process
import pandas as pd
import helper
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import seaborn as sns

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="✨",
    layout="wide",
)

st.sidebar.title("Whatsapp Chat Analyzer ")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = pre_process.preprocess(data)

    # fetch unique users
    user_list = df["user"].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis ", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, words, num_media_messages, total_emojis = helper.fetch_stats(selected_user, df)
        st.title("Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header('Total Messages ')
            st.title(num_messages)

        with col2:
            st.header("Total words ")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Emoji Shared")
            st.title(total_emojis)

        # st.title("Monthly Timeline")
        # timeline =helper.monthly_timeline(selected_user,df)
        # fig,ax = plt.subplots()
        # ax.plot(timeline['time'],timeline['messages'],color="green")
        # plt.xticks(rotation = 'vertical')
        # st.pyplot(fig)



        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)

            # calculate figsize and adjustment of graph size
            num_data_points = len(x.index)
            figsize = (6, 5) if num_data_points >= 5 else (4, 4)
            bar_width = 0.3 if num_data_points >= 4 else 0.01  # Adjust the bar width as needed
            bar_height = 30 if num_data_points >= 5 else 15  # Adjust the bar height as needed

            # Create a black background figure
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor('white')  # Set the background color of the entire figure to black
            ax.set_facecolor('white')

            col1, col2 = st.columns(2)

            with col2:

                # Check if the number of users is greater than 5
                if len(x) > 5:
                    st.title("Graphical Representation")

                    # Set the color of the bars and lines in various shades of white
                    ax.bar(x.index, x.values, color='lavender', edgecolor='black', linewidth=1.5, hatch='',
                           align='center')

                    # Set the color of x-axis and y-axis labels to a contrasting color
                    plt.xticks(rotation=75, color='#333333', size='10')
                    plt.xlabel("Users")
                    plt.yticks(color='#333333', size='10')
                    plt.ylabel("No of Messages")

                    # Set the color of axes lines to a contrasting color
                    ax.spines['bottom'].set_color('#333333')
                    ax.spines['top'].set_color('#333333')
                    ax.spines['right'].set_color('#333333')
                    ax.spines['left'].set_color('#333333')

                    # Set the background color of the entire figure to a slightly off-white
                    fig.patch.set_facecolor('#FAFAFA')

                    st.pyplot(fig)
                else:
                    pass  # Continue with the next code

            with col1:  # code for displaying stats
                st.dataframe(new_df, width=350)

        with col1:

            # Generate the word cloud
            df_wc = helper.create_wordcloud(selected_user, df)

            # Calculate word frequency
            word_freq = pd.Series(df_wc.words_).sort_values(ascending=False)

            # Calculate word usage count
            word_counts = df['messages'].str.split().explode().value_counts().reset_index()
            word_counts.columns = ['Word', 'Count']

            # Merge word frequency and word usage count
            merged_df = word_freq.reset_index().rename(columns={'index': 'Word', 0: 'Frequency'})
            merged_df = merged_df.merge(word_counts, on='Word', how='left').fillna({'Count': 1})

            # Sort by word frequency
            merged_df = merged_df.sort_values(by=['Frequency'], ascending=False).head(50)

            # Rename columns
            merged_df.columns = ['Word', 'Usage Frequency', 'Usage Count']

            # Display the table
            st.title("Most Used Words")
            st.dataframe(merged_df, width=600)




        # Perform linear regression
        model, mae = helper.perform_linear_regression(df)
        rmse, r_squared = helper.calculate_regression_metrics(model, df)

        #i am creating strings for each
        mae_str = f"Mean Absolute Error: {mae:.2f}"
        rmse_str = f"RMSE: {rmse:.2f}"
        r_squared_str = f"R-squared: {r_squared:.2f}"

        # Display results and visualizations
        st.title("Message count prediction per hour")
        st.write(mae_str)
        st.write(rmse_str)
        st.write(r_squared_str)

        # Scatterplot to visualize data and predictions
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=df['hour'], y=df['message_count'], label='Actual Data', linewidth=1)

        # Predict values using the model
        y_pred = model.predict(df[['hour']])

        # Plot the regression line
        sns.lineplot(x=df['hour'], y=y_pred, label='Regression Line', color='red', linewidth=1)

        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Messages')

        plt.legend()

        # Display the plot in the Streamlit app
        st.pyplot(plt)

        # monthly timeline-
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(8, 4))  # Reduced figure size

        # Set the background color to a dark blue shade
        fig.patch.set_facecolor('white')

        # Your plot code here
        ax.plot(timeline['time'], timeline['messages'],linewidth = 1,color="green")  # Use the same orange for the plot
        plt.xticks(rotation="horizontal")
        plt.xlabel("Months")

        # Customize axis labels, colors, etc. as needed
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        plt.ylabel("Number of Messages")
        ax.tick_params(axis='x', colors='black', labelsize=5)
        ax.tick_params(axis='y', colors='black', labelsize=5)
        st.pyplot(fig)

        # daily timeline -
        st.title("Daily Timeline")

        # Call your daily_timeline function
        daily_timeline = helper.daily_timeline(selected_user, df)

        # Get the last 30 days of data
        # last_days_data = daily_timeline.tail(60)

        fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed

        # Set the background color to a dark blue shade
        fig.patch.set_facecolor('white')

        # Plot the last 30 days of data
        ax.plot(daily_timeline['only_date'], daily_timeline['messages'], linewidth = 1,color="green")

        # Customize the x-axis tick labels to show every 5th date for better readability
        # Customize the x-axis tick labels to show every date
        x_tick_labels = daily_timeline['only_date']

        # Set the x-axis tick locations and labels
        ax.set_xticks(x_tick_labels)
        ax.set_xticklabels(x_tick_labels, rotation="horizontal", fontsize=8)  # Adjust rotation and fontsize as needed

        # Customize axis labels, colors, etc. as needed
        ax.xaxis.label.set_color('black')
        plt.xlabel("Days")
        plt.ylabel("Number of Messages")
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black', labelsize=7)
        ax.tick_params(axis='y', colors='black', labelsize=7)

        # Display the plot
        st.pyplot(fig)