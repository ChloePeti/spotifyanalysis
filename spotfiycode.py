import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("bana_471_project_data.csv")

#check all info and view data quality and condition
# print(df.columns)
print(df.info())
# print(df.isnull().sum())
# print('DUPLICATES' , (df["id"].duplicated(keep="first")).sum())

#Drop Duplicates
df = df.drop_duplicates(subset="id", keep="first")

#Drop Null Values
df = df.dropna()

#get rid of songs that are all speech
# df = df[df["speechiness"] <= 0.66]
# R^2 goes from .38 to .41 if we include all speech types

#drop songs longer than a single track length
df = df[df["duration_ms"] <= 360000]

#drop live songs
df = df[df["liveness"] <= .8]

#drop release date
df = df.drop(columns=["release_date"])

#drop old songs
# df = df.loc[df["year"] >= 2001]
# This brings down R^2 significantly

# Define bins for 'popularity' to create a numerical grade column (1 to 10)
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]  # Bins for popularity (0-10, 10-20, ..., 90-101)
grades = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Numerical grades corresponding to the bins

# Use pd.cut() to assign numerical grades based on 'popularity' values
df["grade"] = pd.cut(df["popularity"], bins=bins, labels=grades, right=False)

# Convert 'grade' column to integer
df["grade"] = df["grade"].astype(int)

#regression model
model = smf.ols('popularity ~ tempo + duration_ms + year + danceability + energy + acousticness + explicit + instrumentalness + key + liveness + speechiness + valence', data=df)
reg = model.fit()
print(reg.summary())

#summary stats
pd.set_option('display.max_columns', None)  # Show all columns
selected_vars = ["popularity", "tempo", "loudness", "duration_ms", "year", "danceability", "energy",
                "acousticness", "explicit", "instrumentalness", "key", "liveness", "speechiness", "valence"]
print(df[selected_vars].agg(["mean", "std", "min", "max"]))

# Sort the dataframe by popularity in descending order and get the top 10 songs
top_10_songs = df.sort_values(by="popularity", ascending=False).head(10)

# Display the top 10 songs with relevant columns like track name and popularity
print(top_10_songs[['name', 'popularity']])


df.to_csv('project_data_test2.csv', index=False)
