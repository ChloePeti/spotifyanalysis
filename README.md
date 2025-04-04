#Spotifyanalysis
  
  Using Python and Tableau, my team and I explored and analyzed Spotify song data to understand music industry trends—specifically, what makes a song popular.

My contributions focused on writing and refining the code for data cleaning, performing regression analysis, and generating summary statistics and coefficient tables. Additionally, I designed the final Tableau dashboard to present our findings cohesively.

- <a href="https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks">Dataset Used</a>
- <a href="Spotify Dashboard2.png">View Dashboard</a>
- <a href="Spotify Dashboard1.png">View Additonal Visualizations</a>

INTRODUCTION: 
  By identifying what features of a song make it popular, the music industry can make better data driven decisions to maximize success. Artists and record labels could utilize this insight to write, produce, and market their songs more strategically, ensuring that their releases align with popularity trends. Additionally, streaming services could benefit from this knowledge to help enhance their recommendation algorithms which in turn can help improve user engagement and experience by promoting songs with the highest potential for popularity. 
  In this project, we are analyzing a dataset from Spotify’s Web API covering nearly a century of music from 1921 to 2020. We examined a variety of audio features including the following: danceability, speechiness, acousticness, explicit content, valence, liveness, key, year, energy, duration, tempo, instrumentalness, energy, and year. By examining the relationship between these attributes and a song’s popularity score, we aim to identify the most influential predictors of success.
  
DATA CLEANING:
  To clean the data we removed all rows that proved to have duplicate data, and we did this through using the ID value. Any second occurrence of the same ID was removed from the data set to ensure data integrity and accuracy. We ensured consistency in the data through looking over all of the data in excel, analyzing summary statistics and printing out all of the data types in python. In doing this, we were able to see all the data types and identify any issues within the data. All of the data types were correct as far as we were concerned and none of them needed to be converted for analysis. The only variable that we did chnage was binning the popularity into groups in order to allow an easy transition should we create any graphs to check for outliers and perform other analyses. As for missing data, this was not an issue with this data set. All of the columns had the same number of data points and nothing was obviously out of place. The only column with issues, release year, was already removed so no missing data needed to be handled in this particular data set. As for outliers, there were many decisions to be made here and this is where our initial data clean and the final data clean varied the most. 
  In dealing with outliers in the preliminary data clean, we decided to remove many songs in the data set. Initially, we got rid of all songs over a certain length, songs that had a high speechiness like podcasts, and songs from before the year 2001. Once we performed our regression analysis, the R squared value was extremely low at about .18. We did some alterations to the code to leave these items in as they are not necessarily outliers. The initial thought was to determine what made songs popular in recent years, but this eliminates a lot of the data in relation to this particular popularity data. In the end, we decided to include all years for songs as well as any length or song, but we did remove long songs that were multiple songs or mixed tracks as well as songs that had a high liveness score, meaning they were likely a live performance. This helped to raise the R squared value to .41. In the end, we had 174,389 rows of data of varying types in all rows, meaning there was no missing data, no duplicates, and no extraneous values that do not support our analysis. 

RESULTS:
  Our R-squared Value is 0.415. The R-squared value indicates that 41.5% of the variance in the popularity variable can be explained by the predictors in the model. This means that while the model explains a significant portion of the variability in popularity, there is still a large portion of variability (58.5%) that is unexplained by the current set of predictors.This shows that the model is somewhat effective, but there may be additional factors influencing popularity that are not included in this particular analysis.

Explicit: Explicit content is a strong positive predictor of popularity.
Year: Newer songs are more likely to be popular.
Energy: Lower energy (slower, more mellow tracks) seems to decrease popularity.
Danceability: More danceable songs are associated with higher popularity.
Instrumentalness: More instrumental songs tend to be less popular.
Accousticness: Songs that are more acoustic are generally less popular.
Tempo: For each 1-unit increase in tempo, popularity decreases by 0.0048, holding other variables constant 
Duration (milliseconds): For every additional millisecond in a song popularity score increases by 0.00002656
Key: More instrumental songs tend to have lower popularity decreasing by about 13 per unit increase 
Liveness:  Songs with higher liveness scores tend to be less popular
Speechiness: Songs with more spoken content tend to be significantly less popular
Valance: Songs that are more positive (i.e. higher valance) tend to be more popular. 

 In the end, we found that newer songs are generally more popular. As seen in <a href="Spotify Dashboard1.png">this</a> visual, recently released songs tend to be more popular, even when they are remastered or re-recorded versions. However, the data also reveals a discrepancy in this trend—a cluster of songs from 2000 to 2021 is less popular than expected. While we cannot say for certain, we theorize that this group of outliers may be due to advancements in technology that have enabled more individuals to create and release their own music. Additionally, many factors contribute to a song’s popularity. Generally, songs with higher valence, danceability, and explicit content tend to be more popular. We also observed that more popular songs are less instrumental. Our theory is that this may be because many of today’s popular artists are solo performers rather than bands. We can see overall that the classic bands like The Beatles are still very popular but songs likes drivers license by Olivia Rodrigo and other top rated songs have only one singer. 
