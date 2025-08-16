from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import emoji
import string

extract = URLExtract()

def fetch_stats(selected_user, df):
    """
    Returns: num_messages, total_words, num_media_messages, num_links
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # number of messages
    num_messages = df.shape[0]

    # number of words
    words = []
    if 'message' in df.columns:
        for message in df['message']:
            if isinstance(message, str):
                words.extend(message.split())

    # number of media messages
    num_media_messages = 0
    if 'message' in df.columns:
        num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # number of links
    links = []
    if 'message' in df.columns:
        for message in df['message']:
            if isinstance(message, str):
                links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    """
    Returns:
      x: Series of message counts per user (excludes 'group_notification')
      percent_df: DataFrame with columns ['name', 'percentage'] (index starts at 1)
    """
    # exclude system messages
    mask = df['user'] != 'group_notification' if 'user' in df.columns else pd.Series(dtype=bool)
    x = df[mask]['user'].value_counts()

    # percentage DataFrame
    percent_df = (
        df[mask]['user']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )
    percent_df.columns = ['name', 'percentage']
    percent_df.index = percent_df.index + 1

    return x, percent_df


def create_wordcloud(selected_user, df, stopfile='stop_hinglish.txt'):
    """
    Returns a WordCloud object (can be plotted with ax.imshow(...)).
    If there is no text, returns a WordCloud generated from a single space to avoid errors.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # load stop words
    try:
        with open(stopfile, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except Exception:
        stop_words = set()

    # remove system messages & media
    temp = df[df['user'] != 'group_notification'] if 'user' in df.columns else df.copy()
    if 'message' in temp.columns:
        temp = temp[temp['message'] != '<Media omitted>\n'].copy()
    else:
        temp = temp.copy()

    def remove_stopwords(message):
        if not isinstance(message, str):
            return ''
        y = []
        for word in message.lower().split():
            w = word.translate(str.maketrans('', '', string.punctuation))
            if w and w not in stop_words:
                y.append(w)
        return ' '.join(y)

    wc = WordCloud(width=800, height=600, min_font_size=10, background_color='white')

    if 'message' not in temp.columns or temp['message'].dropna().shape[0] == 0:
        return wc.generate(" ")

    temp['message'] = temp['message'].apply(remove_stopwords)
    corpus = temp['message'].str.cat(sep=" ").strip()
    if corpus == "":
        return wc.generate(" ")
    df_wc = wc.generate(corpus)
    return df_wc


def most_common_words(selected_user, df, stopfile='stop_hinglish.txt', top_n=20):
    """
    Returns a DataFrame with most common words and their counts.
    """
    try:
        with open(stopfile, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except Exception:
        stop_words = set()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # remove system messages & media
    if 'user' in df.columns and 'message' in df.columns:
        temp = df[df['user'] != 'group_notification']
        temp = temp[temp['message'] != '<Media omitted>\n']
    else:
        temp = df.copy()

    words = []
    if 'message' in temp.columns:
        for message in temp['message']:
            if not isinstance(message, str):
                continue
            for word in message.lower().split():
                word = word.translate(str.maketrans('', '', string.punctuation))
                if word and word not in stop_words:
                    words.append(word)

    most_common_df = pd.DataFrame(
        Counter(words).most_common(top_n),
        columns=['word', 'count']
    )
    return most_common_df


def emoji_helper(selected_user, df):
    """
    Returns a DataFrame of emojis and counts.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    if 'message' in df.columns:
        for message in df['message']:
            if not isinstance(message, str):
                continue
            emojis.extend(c for c in message if c in emoji.EMOJI_DATA)

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))),
                            columns=['emoji', 'count'])
    return emoji_df


def monthly_timeline(selected_user, df):
    """
    Returns a DataFrame with columns:
      ['year', 'month_num', 'month', 'message', 'time']
    'time' is like 'Aug 2023' and rows are sorted chronologically.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # ensure we have required columns
    if df.empty or not {'year', 'month_num', 'month', 'message'}.issubset(df.columns):
        return pd.DataFrame(columns=['year', 'month_num', 'month', 'message', 'time'])

    # group and count (use size to avoid counting other columns)
    timeline = (
        df.groupby(['year', 'month_num', 'month'], as_index=False)
          .size()
          .rename(columns={'size': 'message'})
    )

    # rename from size to message if needed (some pandas versions)
    if 'message' not in timeline.columns and 'size' in timeline.columns:
        timeline = timeline.rename(columns={'size': 'message'})

    # sort chronologically
    timeline = timeline.sort_values(['year', 'month_num']).reset_index(drop=True)

    # friendly time label: 3-letter month + space + year
    timeline['time'] = timeline['month'].str.slice(0,3) + ' ' + timeline['year'].astype(int).astype(str)

    return timeline


def daily_timeline(selected_user, df):
    """
    Produce a DataFrame with columns ['date', 'hour', 'message'] where:
      - 'date' is a python date (not datetime)
      - 'hour' is 0..23
      - 'message' is the count of messages for that date+hour
    This function is robust to input columns: prefers 'message_date', then 'date', then 'only_date'.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if df.empty:
        return pd.DataFrame(columns=['date', 'hour', 'message'])

    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # choose best datetime source available
    if 'message_date' in df.columns:
        dt_col = 'message_date'
    elif 'date' in df.columns:
        dt_col = 'date'
    elif 'only_date' in df.columns:
        dt_col = 'only_date'
    else:
        return pd.DataFrame(columns=['date', 'hour', 'message'])

    # If we have a full datetime (message_date or date), extract date and hour
    if dt_col in ['message_date', 'date']:
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        df = df.dropna(subset=[dt_col])
        df['date_only'] = df[dt_col].dt.date
        df['hour'] = df[dt_col].dt.hour
    else:
        # only_date is a date (no hour info). try to use existing 'hour' column if present; else default 0.
        df['date_only'] = pd.to_datetime(df['only_date']).dt.date
        if 'hour' not in df.columns or df['hour'].isnull().all():
            df['hour'] = 0
        else:
            df['hour'] = df['hour'].fillna(0).astype(int)

    # Group and count messages (use size to avoid counting other columns)
    timeline = df.groupby(['date_only', 'hour'], as_index=False).size().rename(columns={'size': 'message'})

    # normalize column name to 'date'
    timeline = timeline.rename(columns={'date_only': 'date'})

    # ensure proper ordering
    timeline = timeline.sort_values(['date', 'hour']).reset_index(drop=True)

    return timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    """
    Returns pivot table indexed by day_name and columns=period counting messages.
    If required columns are missing, returns an empty DataFrame.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    required = {'day_name', 'period', 'message'}
    if not required.issubset(set(df.columns)):
        # return an empty DataFrame with expected shape/labels to avoid breaking callers
        return pd.DataFrame()

    user_heatmap = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)

    return user_heatmap


# Compatibility wrapper / alias so app.py can import `user_heatmap`
def user_heatmap(selected_user, df):
    return activity_heatmap(selected_user, df)


def heatmap_params(selected_user, df, cmap='RdYlGn'):
    """
    Return (user_heatmap_df, cmap_name, vmin, vmax)

    - cmap default 'RdYlGn' maps low -> red, mid -> yellow, high -> green (red-to-green).
    - vmin and vmax are numeric min/max values of the pivot table; they are None if the pivot is empty
      or could not be converted to numeric.
    """
    uh = activity_heatmap(selected_user, df)
    if uh is None or uh.empty:
        return uh, cmap, None, None

    # try to compute numeric min/max for vmin/vmax; if conversion fails, leave them None
    try:
        numeric = uh.astype(float)
        vmin = float(numeric.min().min())
        vmax = float(numeric.max().max())
    except Exception:
        vmin = None
        vmax = None

    return uh, cmap, vmin, vmax
