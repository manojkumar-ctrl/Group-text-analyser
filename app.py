# app.py
import streamlit as st
import pandas as pd
import re
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar Title
st.sidebar.title('Whatsapp Chat Analyser')

# File Uploader
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    # try decode in utf-8, fallback to latin1
    try:
        data = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        data = bytes_data.decode("latin1")

    # Preprocess chat data
    df = preprocessor.preprocessor(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    # User selection
    selected_user = st.sidebar.selectbox("Show analysis with respect to : ", user_list)

    # Analysis button
    if st.sidebar.button("Show Analysis"):
        # Fetch basic stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.header("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.header(num_messages)
        with col2:
            st.header("Total Words")
            st.header(words)
        with col3:
            st.header("Total Media Shared")
            st.header(num_media_messages)
        with col4:
            st.header("Total Links Shared")
            st.header(num_links)

        # Timeline for chat
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        if not timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        else:
            st.info("No timeline data to display.")


        # daily timeline
        # Daily Timeline — line graph (no pointed dots)
        st.title("Daily Timeline (Messages per Day)")

        daily_timeline = helper.daily_timeline(selected_user, df)

        if not daily_timeline.empty:
            # aggregate hourly counts into per-day counts
            daily_counts = daily_timeline.groupby('date', as_index=False)['message'].sum()

            # ensure 'date' is datetime for plotting
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])

            # optional: compute a 7-day rolling average for smoothing
            daily_counts = daily_counts.sort_values('date').reset_index(drop=True)
            daily_counts['rolling_7d'] = daily_counts['message'].rolling(window=7, min_periods=1, center=False).mean()

            fig, ax = plt.subplots(figsize=(12, 5))
            # main line (no markers)
            ax.plot(daily_counts['date'], daily_counts['message'], linewidth=2, marker=None, label='Messages per day')
            # optional smoothed line (dashed)
            ax.plot(daily_counts['date'], daily_counts['rolling_7d'], linewidth=1.5, linestyle='--', label='7-day avg')
            # subtle filled area under the main line
            ax.fill_between(daily_counts['date'], daily_counts['message'], alpha=0.1)

            ax.set_xlabel("Date")
            ax.set_ylabel("Number of messages")
            ax.set_title("Daily messages")
            ax.legend()
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No timeline data to display.")



        # activity map
        st.title("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)
        with col2:
            st.header("Most Busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # --- Heatmap: create Figure and Axes, draw heatmap onto Axes, pass Figure to st.pyplot ---
        # get pivot + params from helper
        user_heatmap, cmap, vmin, vmax = helper.heatmap_params(selected_user, df)

        if user_heatmap is not None and not user_heatmap.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(user_heatmap, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=False)
            ax.set_title("Activity Heatmap (red -> green)")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No heatmap data to display.")

        # Most Busy Users (only for Overall)
        if selected_user == "Overall":
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)

            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='r')
                plt.xticks(rotation=90)
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title('Word Cloud')
        df_wc = helper.create_wordcloud(selected_user, df)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(df_wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)



        # Most Common Words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user, df)

        if not most_common_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(most_common_df['word'], most_common_df['count'])
            plt.xticks(rotation=90)
            st.pyplot(fig)
        else:
            st.info("No words to display.")

        # Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(
                    emoji_df['count'].head(10),
                    labels=emoji_df['emoji'].head(10),
                    autopct="%0.2f"
                )
                st.pyplot(fig)
            else:
                st.info("No emojis to display.")
