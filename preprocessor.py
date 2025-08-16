# preprocessor.py
import re
import pandas as pd

def preprocessor(data):
    """
    Parses a WhatsApp chat export into a DataFrame with:
    columns: user, message, message_date, date, year, month_num, month, day, hour, minute, only_date
    Future-dated messages (dates > now) are dropped to avoid impossible timeline entries.
    """
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[ap]m)?\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # keep message_date column (datetime) for compatibility with helper functions
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # clean and parse message_date into datetime
    df['message_date'] = df['message_date'].astype(str).str.rstrip(' -').str.strip()
    df['message_date'] = pd.to_datetime(df['message_date'], dayfirst=True, errors='coerce')

    # split user and message text
    users = []
    messages_text = []
    for message in df['user_message']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)
        if len(entry) > 2 and entry[1].strip():
            users.append(entry[1].strip())
            messages_text.append(entry[2])
        else:
            users.append('group_notification')
            messages_text.append(entry[0])

    df['user'] = users
    df['message'] = messages_text
    df.drop(columns=['user_message'], inplace=True)

    # Create 'date' column (helper functions expect df['date'])
    df['date'] = df['message_date']

    # Drop rows with parsing failures (NaT)
    df = df.dropna(subset=['date']).reset_index(drop=True)

    # Remove future-dated messages (dates > now)
    now = pd.Timestamp.now()
    future_mask = df['date'] > now
    if future_mask.any():
        df = df.loc[~future_mask].reset_index(drop=True)

    # Extract time features after removing bad/future dates
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name']=df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    # NOTE: do NOT drop 'message_date' -- keep it for backward compatibility
    return df
