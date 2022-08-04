# song-dataset-prediction
This project was completed in Spring 2022 for ISYE 7406 Data Mining and Statistical Learning, a graduate course through Georgia Tech. 

Predict year song was released, kmeans clustering, TF-IDF. See PDF for full analysis and writeup. 

Uses 10,000 song file subset of the One Million Song Dataset (MSD). Years of song predictions were made using multiple different models. Kmeans was used to determine if clustering naturally grouped songs into periods or decades. TF-IDF was used to identify high-value words by decade. 

No push requests accepted

# Abstract

Using a subset of the Million Song Dataset (MSD), I try and predict the year a song was released based on features like the average and covariance timbre values for each song, song duration and loudness, and others. Models built include linear regression, ridge regression, LASSO, random forest, boosting, and others to predict the response, year. 

I also used k-means to cluster the songs to understand if, based on the chosen features, the songs are naturally grouped into time periods.

Finally, I use natural language processing (NLP), specifically term frequency-inverse document frequency (TF-IDF) to find the words that were most indicative of a song from a particular decade. 

# Introduction

The problem of predicting song year is an important one for marketing and music companies, so it is beneficial to have models that can do this accurately. One reason the year is important is that a consumer of a certain age may like songs they grew up with and be more interested in purchasing or listening to songs from that time period. It makes sense to advertise or play these songs if you believe the consumer may want to purchase or listen to them. This type of advertising and playlist building can improve a company’s profits, either directly through sales or by driving more users to a music platform which can lead to new subscribers or improved advertising opportunities. 

Year is also important since it can also be a proxy for music type. If music details are not readily available, but other songs a consumer listens to are from a genre that’s strongly associated with a year then it may make sense to advertise or play songs from said year.

Challenges of building models to predict songs’ years are multiple. One problem is the amount of data means that a lot of processing power is required. Using song elements like pitch and timbre leads to a dataset with lots of features. In addition, models should be trained with lots of observations. Some ways to address these issues are to use smaller datasets, cloud-computing, or to use principal component analysis to reduce the dimension being worked in. 

Another challenge is that music doesn’t often change in a significant way from one year to the next which makes predictions more difficult. One option to address this is to use ensemble methods to improve predictive power.

# Data Sources and Exploratory Data Analysis

The Million Song Dataset is a dataset with one million songs that was created as a collaboration between the Laboratory for the Recognition and Organization of Speech and Audio (LabROSA) and The Echo Nest. The Echo Nest has since been purchased by Spotify.  

Initially, I downloaded the MSD dataset from the UCI Machine Learning Repository. From the one million songs in the dataset, it contained 515,345 observations (songs) since that was the number of songs that contained the year (the response variable). This dataset had been processed so there was 1 response variable (year) and 90 predictor variables. The predictor variables were the 12 averages of the 12-dimensional timbre values and the 78 associated timbre covariances.

This dataset didn't include song names (for NLP analysis), so I decided to use a subset of 10,000 song files from the Million Song Dataset website. This was approximately 2GB in size in a compressed format, as downloaded, and over 2.5GB after extraction. I had to read in the features from each individual file to create my own dataset which was smaller than the dataset from UCI and which contained song names.

Using my dataset created from 10,000 songs, I inspected the dataset and found that 4680 songs contained a year, so I kept only these songs. The earliest year was 1926 and the most recent song was from 2010. There were 112 predictor variables and one response variable, year.

To inspect the distribution of songs, I assigned each song a “decade” classification based on the year the song was released and found that most songs were from the 1990s and 2000s with very few songs from the 1950s and before. The distribution is heavily skewed towards songs released more recently. 

Based on this distribution, I decided that since there were so few songs from pre-1950, I would not use those songs in my dataset. Although I could have used these songs to predict the year, I wanted to use the same dataset for all portions of my analysis. Since the number of songs (and thus, the number of terms in the song titles) was very small for the 1920s, 1930s, and 1940s, this meant that my natural language processing results would have been determined by only these few songs and wouldn’t have been meaningful. Although the 2010 decade only contains songs from a single year, I decided to keep it because there were more songs (64 total songs from 2010). 

Due to this imbalance and because I was most interested in predicting the year (rather than decade), I decided that it made the most sense to treat the first part of my analysis as a regression problem rather than a classification problem. As an example, predicting a song that is actually from 1980 as being from 1979 would misclassify the song if using a decade classification even though a prediction error of one year would be good. Similarly, if the song was predicted to be from 1989, it would be classified as correct if using decades even though the error is nine years so is far worse than a one-year error if 1979 was predicted. 

# Proposed Methodology
My methodology begins with data gathering and pre-processing. 

First, as described previously, to get a more complete dataset than available from UCI (including song names and other details), I had to download a 10,000-song subset from the Million Song Database website. 

For each of the 10,000 song files, I read the song information and metadata which included numerous features. 

I noticed that the timbre average and covariance values from the UCI dataset were not provided in that format in the song files I downloaded; instead, timbre values were provided by segments throughout the duration of the song, so I had to compute my own timbre average and covariance values. 

After gathering and cleaning this data, I wrote the dataset to a CSV file for easier processing moving forward. Using this data, I further cleaned up the songs and features I intended to use as discussed in the previous section.

Even after preprocessing to minimize the number of songs and features, I still had 4664 songs (observations) and 98 features. It can be computationally expensive to build cross-validated models with this many features, but regardless, I decided to build the following models. Before I started building models, I also scaled the predictor variables.

1. Linear Regression
2. Stepwise Regression (AIC)
3. Ridge
4. LASSO
5. Partial Least Squares (PLS)
6. Random Forest
7. Boosting

Year Prediction

I decided to use a 75/25% train/test split, and I used cross-validation to determine the best model and arrive at the most robust conclusion possible. For each iteration (with different train/test subsets), I recorded the test errors for each model and to come up with a final test error rate I averaged across all iterations. When training each model, I used the same random training/testing split for all models for a given iteration. 

Because I expected the process to take a long time to run, I experimented by running only five iterations to start. Even only performing five train/test iterations, it took almost 30 minutes to run my code and predict with the seven above models. As a result, I decided to only run 30 iterations.

Linear and Stepwise Regression are straightforward in that there are no tuning parameters to consider when building these models. For Ridge and LASSO, for each train/test split, the optimal lambda value was chosen to ensure the best model was used when making predictions. 

For PLS and Boosting, I used 10-fold cross-validation within each iteration. Since I am using Boosting for regression and not classification, I used a Gaussian distribution with 5000 trees. 

For Random Forest, I used 32 random predictor variables (using a p/3 rule of thumb, where p = 97) as candidates for each split and grew 500 trees.

All models were built 30 times with different train/test splits as detailed above and predictions were made for all songs in the test subset for each model. Each calculated test error is the mean squared error (MSE) for the test predictions when compared to the true year that the song was released. The overall test error rate for each model was calculated as the average of all 30 individual MSE test error rates. The model with the lowest MSE will be deemed the best model, with possible consideration for the amount of variance. 

Using cross-validation provides a much more robust and accurate overall test error rate since there is no concern that any given random split provided an outlier result where the test error rate was unusually high or low. 

Clustering

When using k-means to cluster the songs. I built models with up to 15 clusters, and I used the silhouette method (see Appendix for discussion) to determine the optimal number of clusters. Initially, I attempted to use the elbow method, but the optimal number of clusters wasn’t clear in my plot which is why I used the silhouette method. I visualized the resulting clusters, and the results are detailed in the following section. Using the results, I will check to see if any groupings by year are evident. 

Natural Language Processing

To determine what keywords are most indicative of a given decade, I used Term Frequency-Inverse Document Frequency (TF-IDF). This method looks at how common a term (a single word from a song title in this case) is in the overall corpus (all the song titles) as well as how frequently it appears in a certain class (decade) and computes a score to show how indicative the word is of a particular class. To compute the scores, I first tokenize (separate all the individual words from the song title) and remove stop words. Stop words are generally common words like “the, and, in, on,” that are not informative when looking at different classes. 

# Analysis and Results

All analysis was done using R as a programming language and RStudio software. 

Year Prediction

When predicting the year that a song was released, the seven different models provided the below mean squared errors (MSE). As shown, boosting provided the smallest test error and can be considered the best model. However, it had the worst variance. 

If the amount of variance in the boosting model was considered a problem, then the second-best model was Random Forest, and it also had the smallest variance so it would be the best choice. 

Although boosting had the worst variance, it was similar to some of the other models and because it outperformed all other models for MSE by quite a lot, I still consider boosting to be the best model for predicting song years with this dataset. 

The other five models had similar test errors to each other, but of those models LASSO was the best and Stepwise Regression with AIC was the worst. 

Test Error (MSE)	Variance
1. Linear Regression	        100.19	23.76
2. Stepwise Regression (AIC)	100.49	24.61
3. Ridge Regression	        99.89	23.95
4. LASSO	99.26	22.84
5. Partial Least Squares (PLS)	100.19	23.80
6. Random Forest	95.34	21.23
7. Boosting	89.90	25.15

| Model      | Test Error (MSE) | Variance     |
| :---        |    :----:   |          ---: |
| Linear Regression      | 100.19       | 23.76   |
| Stepwise Regression (AIC)   | 100.49        | 24.61      |
| Ridge Regression       |  99.89       | 23.95   |
| LASSO       | 99.26       | 22.84   |
| Partial Least Squares (PLS)      | 100.19       | 23.80   |
| Random Forest      | 95.34       | 21.23   |
| Boosting       | 89.90       | 25.15   |

Clustering

For K-means, I calculated the Within-Cluster-Sum-of-Squared-Errors and plotted them to find the optimal number of clusters. It was difficult to visually choose the optimal number of clusters from this plot using the elbow method because the values at k=3 and k=4 didn’t create a smooth plot. As such, I used the silhouette method and found the optimal number of clusters was k=2.

Using k=2 clusters, I wanted to understand if these two clusters separated the songs by some year (e.g., before and after 1990), so I created histograms and a cluster plot. 

There are differences in the histograms in the number of observations by year, but there isn’t a clear separation. In the cluster plot (which uses the top two principal components since there are more than two features), there is a lot of overlap which would further suggest that the songs are not distinct and well separated in the two clusters. As a result, clustering does not seem to be a good way to separate songs by year with the data that was gathered. 

Natural Language Processing

Using TF-IDF, I found the top 10 words per decade. For the 1950s and 2010s, there were relatively few songs and as a result there were a lot of ties, which meant that some top words only appeared in the decade once. As a result, there were more than 10 songs for these decades. 

A selection of interesting “Top 10” words is provided below. 

| Decade      | Top Words  |
| :---          |          ---: |
|1950s | carnaval, cantando, conguitos, faith|
|1960s | watermelon, quiet, spoken, word, lp|
|1970s | bit, bully, version, digital, remaster, lp, waste|
|1980s | start, version, make, night, Halloween, windpower|
|1990s | version, lp, live, explicit, red, album, mix|
|2000s | remix, version, live, amended, album, explicit, featuring, edit, mix, club|
|2010s | feat, diamond, cannonball, minion, twins, underworld|

As shown, there were several words that made the Top 10 list even though they were important in more than one decade. Some examples are: 

version, lp, album, explicit

I decided not to add these to my stop words list because I thought it was interesting to see how these terms change over time. For example, we start seeing the words “explicit” and “live” in song titles in the 1990s, which seems reasonable if you listened to alternative rock in the late 1990s. Songs also started “mix”-ing in the 1990s and “remix”-ing in the 2000s. 

Similarly, we start seeing the words “featuring” and “feat” in the 2000s and 2010s. Although artists have collaborated for a long time, it seems like song titles regularly started saying that the song featured another artist in the 2000s. And on the other hand, the term “lp” is no longer important after the 1990s as might be anticipated. 

Notably, several of the top words from the 1950s are in other languages, whereas this is less common in other decades. And the 1970s made use of terms like “bit,” “digital,” and “remaster,” as music moved from analog to digital. 

Many of these observations from different decades for high-value words are understandable if you lived through or understand music in these decades.

For fun, some surprising terms are “watermelon” which was tied for the more important word in the 1960s. “Bully” and “waste” were top 5 words in the 1970s, while “Halloween” and “windpower” were top 10 words in the 1980s and “club” was a top word in the 2000s (not overly surprising). The 2010s had a lot of ties, so although many top words only appeared once, it’s entertaining to see “minion,” “twins,” and “underworld” tied for top words even if these may not be very meaningful due to the small number of songs. 

# Conclusion

In conclusion, boosting provided the best model with the lowest test error rate. The second-best model was random forest which also had the smallest variance, as opposed to boosting which had the worst variance. 

However, the best prediction score was an MSE of 89.9, which means the average error was approximately 9.5 years. One possible reason the test error wasn’t better may be because music often changes incrementally. As such, it is difficult to predict what year a song is from with a high level of accuracy. 

Using K-means clustering didn’t provide a meaningful separation of songs, at least not that was correlated with year. If other classifications were available, it might have been interesting to see if clusters corresponded to something else. 

Natural Language Processing provided some interesting high-value words, in some cases allowing us to draw conclusions about music trends over time. 

Possible future work includes gathering more complete information for some discarded features to see if they provide additional predictive power. For example, if Latitude and Longitude details were available for all artists and songs, then if music was coming out of a small region or similar area during a particular time period, this might improve the predictive power of the models. 

More complete information could also include gathering genre classifications for all songs and seeing if these lead to improved year predictions. K-means clustering models could also be updated to see if clusters indicate time periods once genre is included. Alternately, models could be built that included year and then compared to genre labels to see if clustering is indicative of genre. 

One final future project could be to predict the years for all songs in an album and find the mean or median of these songs to see if this leads to improved prediction. 
