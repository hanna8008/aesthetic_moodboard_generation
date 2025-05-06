import pandas as pd
df = pd.read_csv('data/filtered_mood_color_dataset.csv')
print(f'Total entries: {len(df)}')
print(f'Mood-labeled entries: {df[\"mood\"].notna().sum()}')
