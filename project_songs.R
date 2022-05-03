#Project
library(rhdf5)
library(stats)
library(dplyr)
library(tidyverse)
library(mltools)
library(tokenizers)
library(tidytext)
library(leaps)
library(MASS)
library(lars)
library(pls)
library(randomForest)
library(gbm)
library(corrplot)
library(RColorBrewer)
library(lares)
library(factoextra)
library(tm)

set.seed(1)

#read in data from UCI MillionSong Dataset
#only contains song year, timbre averages and covariance. Not song or artist name, etc.
data1 <- read.table("YearPredictionMSD.txt", sep = ',')

########
#read in .h5 data files from MillionSong Dataset website
#read in for subset of 10,000 songs only, not all 1M to keep computation size reasonable
#example:
#filename = "TRAAABD128F429CF47.h5"

######################################################################
#Get required data from .h5 files (for 10000 songs)
#After running once, can import the generated .csv rather than rerun 
#this section of code to get the data
######################################################################

#get list of files for 10000 songs
vec_of_files <- list.files(getwd(), pattern = '.h5',
                           full.names = F)

fulldata <- NULL

#for loop to get data from all songs
for (filename in vec_of_files){
  temp1 = h5read(filename, 'analysis')
  temp2 = h5read(filename, 'metadata')
  temp3 = h5read(filename, 'musicbrainz')
  
  #start adding info to 'data'
  data <- temp2$songs
  data$year <- temp3$songs$year

  #get average timbre and covariance for each of 12 timbre categories
  avgtimbre = NULL
  covar = NULL
  for (i in 1:12){
    avgtimbre <- cbind(avgtimbre, mean(temp1$segments_timbre[i,]))
    for (k in i:12){
      covwith <- temp1$segments_timbre[k,]
      covar <- cbind(covar, cov(temp1$segments_timbre[i,], covwith))
    } #end of k for loop
  } #end of i for loop
  
  #bind data for this song to 'data'
  data <- cbind(data,avgtimbre, covar)
  data$duration <- temp1$songs$duration
  data$end_of_fade_in <- temp1$songs$end_of_fade_in
  data$loudness <- temp1$songs$loudness
  data$start_of_fade_out <- temp1$songs$start_of_fade_out
  data$tempo<- temp1$songs$tempo
  data$track_id <-temp1$songs$track_id
  #bind data for all songs to 'fulldata'
  fulldata <- rbind(fulldata, data)

} #end of main for loop for each song file

fulldata <- subset(fulldata, select = -c(genre, idx_artist_terms, idx_similar_artists, artist_playmeid, analyzer_version))

#get and write column names to the dataframe
avgtimbre_col <- c('avgtimbre1','avgtimbre2','avgtimbre3','avgtimbre4','avgtimbre5','avgtimbre6','avgtimbre7','avgtimbre8','avgtimbre9','avgtimbre10','avgtimbre11','avgtimbre12')
covar_col <- c("cov1_1", "cov1_2", "cov1_3", "cov1_4","cov1_5",  "cov1_6", "cov1_7", "cov1_8",
               "cov1_9", "cov1_10",   "cov1_11",  "cov1_12",  "cov2_2", "cov2_3", "cov2_4", "cov2_5", 
               "cov2_6",  "cov2_7",  "cov2_8","cov2_9",  "cov2_10",   "cov2_11",  "cov2_12", "cov3_3",
               "cov3_4",  "cov3_5",   "cov3_6", "cov3_7",    "cov3_8","cov3_9",  "cov3_10",  "cov3_11", "cov3_12",
               "cov4_4",   "cov4_5",   "cov4_6", "cov4_7",  "cov4_8","cov4_9",  "cov4_10",   "cov4_11",  "cov4_12",
               "cov5_5", "cov5_6",  "cov5_7",  "cov5_8", "cov5_9", "cov5_10",   "cov5_11",   "cov5_12",  "cov6_6", 
               "cov6_7",   "cov6_8", "cov6_9", "cov6_10",  "cov6_11",   "cov6_12", "cov7_7",
               "cov7_8", "cov7_9",   "cov7_10", "cov7_11",  "cov7_12",  "cov8_8",   "cov8_9", "cov8_10",
               "cov8_11", "cov8_12",   "cov9_9","cov9_10",  "cov9_11",  "cov9_12",  "cov10_10",  "cov10_11",
               "cov10_12", "cov11_11", "cov11_12",  "cov12_12")
names(fulldata)[17:28] <-avgtimbre_col
names(fulldata)[29:106] <-covar_col

write.csv(fulldata, file = "fulldata.csv")
###############################################################################
#End of code to get required data and create .csv
#Can start below once the .csv is generated. No need to rerun above code section
###############################################################################

csvdata <- read.csv("fulldata.csv", header = TRUE)

###############################################################################
#Continue preparing data whether from .csv file or just read from .h5 files

data <- subset(csvdata, year != 0) #get rid of observations without a listed year
data <- subset(data, year >=1950) #start from 1950 due to very few songs in 1920s, 30s, 40s
data <- subset(data, select = -c(artist_7digitalid, artist_id, Index, song_hotttnesss, artist_location, artist_latitude, artist_longitude, song_id, track_id, artist_mbid, release_7digitalid, track_7digitalid))

data <- data %>% 
  mutate(decade = case_when(year < 1930 ~ 1920,
                            year < 1940 ~ 1930,
                            year < 1950 ~ 1940,
                            year < 1960 ~ 1950,
                            year < 1970 ~ 1960,
                            year < 1980 ~ 1970,
                            year < 1990 ~ 1980,
                            year < 2000 ~ 1990,
                            year < 2010 ~ 2000,
                            year >= 2010 ~ 2010))

###################################################################
#
#Start of cross validation for predictions and MSE calcs
#
##################################################################

### save the TE values for all models in all $B=100$ loops
B= 30; ### number of loops
TEALL = NULL; ### Final TE values
te1<- te2<-te3<- te4<-te5<- te6 <- te7 <- te8 <- te9 <- c()
n = length(data$year)
n1 = round(n*.25)  #get 25% for testing

data_sub = subset(data, select = -c(decade,title,release,artist_name))
#scaling predictor variables
temp = data_sub$year
data_sub = as.data.frame(scale(subset(data_sub, select = -c(year))))
data_sub$year = temp

####Additional exploratory plots before continuing
######################################
hist(data_sub$year, xlab = "Year", main = "Histogram of Songs by Year (Subset of data)")

res <- round(cor(data_sub),2);
#corr plot between all variables
corrplot::corrplot(res, type="upper", order="hclust",
                   col=brewer.pal(n=8, name="RdYlBu"))
#corr chart with top 10 paired variables
corr_cross(data_sub, # name of dataset
           max_pvalue = 0.05, # display significant correlations only
           top = 10) # top 10 pairs of variables by corr coeff
#corr chart with top 10 variables correlated with response, V1
corr_var(data_sub,
         year, # response var
         max_pvalue = 0.05,
         top = 10) # top 10

#Start of loop for CV model building, predictions
###############################################################
for (b in 1:B){
  ### randomly select n1 observations as testing data in each loop
  flag <- sort(sample(1:n, n1));
  datatrain_sub = data_sub[-flag,];
  datatest_sub = data_sub[flag,];
  ytrue = datatest_sub$year
  
  datatest_sub = subset(datatest_sub, select = -c(year))
  
  ### Suppose that you save the TE values for these models as
  ### te1, te2, te3, te4, te5, te6, te7 respectively, within this loop

  #1 linear reg w/ all predictors (not decade, title, release, artist name)
  model1 <- lm(year~., data = datatrain_sub)
  
  # Model 1: testing error 
  ypred1test <- predict(model1, newdata = datatest_sub);
  te1 <-   mean((ypred1test - ytrue)^2);
  
  #3 Linear regression with variables (stepwise) selected using AIC
  model3 <- step(model1, k=2, trace = FALSE)  #k=2 to get AIC
  ## Model3 Testing errors 
  ypred3test <- predict(model3, datatest_sub);
  te3 <-   mean((ypred3test - ytrue)^2);
  
  #4 Ridge Reg
  model4 <- lm.ridge(year ~ ., data = datatrain_sub, lambda= seq(0,100,0.001))
  ## Find the "index" for the optimal lambda value for Ridge regression 
  ##        and auto-compute the corresponding train and test error 
  indexopt <-  which.min(model4$GCV);  
  ridge.coeffs = model4$coef[,indexopt]/ model4$scales;
  intercept = -sum( ridge.coeffs  * colMeans(subset(datatrain_sub, select = -c(year)))  )+ mean(datatrain_sub$year);
  
  ## Model 4 (Ridge):  testing errors in the test set 
  ypred4test <- as.matrix(datatest_sub) %*% as.vector(ridge.coeffs) + intercept;
  te4 <-  mean((ypred4test - ytrue)^2); 

  #5 LASSO
  model5 <- lars( as.matrix(subset(datatrain_sub, select = -c(year))), datatrain_sub$year, type= "lasso", trace= FALSE);
  ## Choose the optimal lambda value that minimizes Mellon's Cp criterion 
  Cp1  <- summary(model5)$Cp;
  index1 <- which.min(Cp1);
  
  lasso.lambda <- model5$lambda[index1]
  
  ## Model 5:  testing error for lasso  
  pred5test <- predict(model5, as.matrix(datatest_sub), s=lasso.lambda, type="fit", mode="lambda");
  ypred5test <- pred5test$fit; 
  te5 <- mean( (ypred5test - ytrue)^2); 
  
  #7 Partial Least Squares
  model7 <- plsr(year ~ ., data = datatrain_sub, validation="CV");
  
  ### 7(i) auto-select the optimal # of components of PLS 
  ## choose the optimal # of components  
  mod7ncompopt <- which.min(model7$validation$adj);
  ## The opt # of components, it turns out to be 17 for this dataset,
  ##       and thus PLS also reduces to the full model   
  
  ## 7(iii) Testing Error with the optimal choice of "mod7ncompopt" 
  ypred7test <- predict(model7, ncomp = mod7ncompopt, newdata = datatest_sub); 
  te7 <- mean( (ypred7test - ytrue)^2); 
  
  #8 Random Forest
  model8 <- randomForest(year ~., data=datatrain_sub, 
                         importance=TRUE)
  ypred8test = as.numeric(predict(model8, datatest_sub, type='class'))
  te8 <- mean( (ypred8test - ytrue)^2); 
  ###
  
  #9 Boosting
  gbmforest <- gbm(year ~ .,data=datatrain_sub,
                   distribution = 'gaussian',
                   n.trees = 5000, 
                   shrinkage = 0.01, 
                   interaction.depth = 3,
                   cv.folds = 10)
  
  ## Model Inspection 
  ## Find the estimated optimal number of iterations
  perf_gbm1 = gbm.perf(gbmforest, method="cv") 
  #find test error
  ypred9test <- predict(gbmforest,newdata = datatest_sub, n.trees=perf_gbm1)
  te9 <- mean( (ypred9test - ytrue)^2); 
  
  TEALL = rbind( TEALL, cbind(te1, te2, te3, te4, te5, te6, te7, te8, te9) );
}
dim(TEALL); ### This should be a Bx7 matrices
### if you want, you can change the column name of TEALL
#colnames(TEALL) <- c("mod1 (Lin Reg)", "mod2 (Lin Reg, best k=5)", "mod3 (Lin Reg, step AIC)", "mod4 (Ridge Reg)", "mod5 (LASSO)", "mod6 (PCR)", "mod7(PLS)");
colnames(TEALL) <- c("Linear Reg", "Stepwise (AIC)", "Ridge Reg", "LASSO", "PLS", "Random Forest", "Boosting");
## You can report the sample mean and sample variances for the seven models

TEmeans <- apply(TEALL, 2, mean);
TEvar <- apply(TEALL, 2, var);
TEmeans
TEvar

par(mar = c(8,5,4,3))
plot(TEmeans, type = "b",col="red", ylab="Test Error Rate (MSE)", xlab="", lwd=2, font.lab=2, main="Model Comparision (MSE)",
     cex.axis = 1.5,cex.lab = 1.5,font=2, cex.main=1.5, xaxt = 'n')
axis(1, at = 1:7, labels = colnames(TEALL), las = 3)
par(mar = c(8,5,4,3))
plot(TEvar, type = "b",col="blue", ylab="Test Error Rate (Var)", xlab="", lwd=2, font.lab=2, main="Model Comparision (Variance)",
     cex.axis = 1.5,cex.lab = 1.5,font=2, cex.main=1.5, xaxt = 'n')
axis(1, at = 1:7, labels = colnames(TEALL), las = 3)

colnames(data_sub)


######################################################################
#
#End of cross validation for predicted values and MSE calcs
#
######################################################################
#
# Start of Kmeans analysis
#
####################################
library(ggplot2)
library(factoextra)

cluster_num1 <- c(15)
clusterdetails <- compactness <- c(15)

#loop to check different number of centers (1 to 15)
for (k in 1:15){
  cluster1 <- kmeans(x= subset(data_sub, select = -c(year)), centers = k, nstart = 25, iter.max = 50) #using all pred
  #cluster2 <- kmeans(x= subset(data_sub, select = -c(year)), centers = k, nstart = 25) #using column subset
  cluster_num1[k] <- cluster1$withinss
  compactness[k] <- cluster1$betweenss/cluster1$totss #percent the represents compactness of clustering
  clusterdetails[k] <- cluster1
  if (k == 2){
    temp <- cluster1
  }
}
#plot tot.withinss for diff number of centers and diff predictors. Tot.withinss is good metric for finding elbow point
#can try a ratio with betweenss/totss also

plot(cluster_num1, main = 'K-means', xlab = 'Number of Clusters (Centers)', ylab= 'Within Cluster Sum of Squares', col = 'red')  #using all pred
#The plot shows a big change from 3 to 4 and then the change is less, so it looks like k=4 clusters is the right choice
#or k=6, for one run. Changes by random seed and how the centers are selected initially
plot(compactness, main = 'K-means Compactness', xlab = 'Number of Clusters (Centers)', col = 'blue', ylab= 'Clustering Compactness (Percent)') #plot compactness by number of clusters

fviz_nbclust(x = subset(data_sub, select = -c(year)), kmeans, method = 'silhouette')

selectk = 2
#after picking k value, show the indices and which cluster each observation was assigned to

yearbycluster <- list()

#get the years for observations in each cluster (e.g. cluster 3 or 4)
for (i in 1:selectk){
  idx = ifelse(clusterdetails[[selectk]]==i,TRUE, FALSE)
  yearbycluster[[length(yearbycluster)+1]] <- data_sub[idx,]$year
}

par(mfrow = c(1,2))
hist(yearbycluster[[1]], main = "Histogram for Years of Cluster 1", xlab = "Year")
hist(yearbycluster[[2]], main = "Histogram for Years of Cluster 2", xlab = "Year")
#hist(yearbycluster[[3]], main = "Histogram for Years of Cluster 3", xlab = "Year")
#hist(yearbycluster[[4]], main = "Histogram for Years of Cluster 4", xlab = "Year")
#hist(yearbycluster[[5]], main = "Histogram for Years of Cluster 5", xlab = "Year")
#hist(yearbycluster[[6]], main = "Histogram for Years of Cluster 6", xlab = "Year")
#hist(yearbycluster[[7]], main = "Histogram for Years of Cluster 7", xlab = "Year")
#hist(yearbycluster[[8]], main = "Histogram for Years of Cluster 8", xlab = "Year")

#cluster plot for k=2, based on this being the optimal number of clusters
fviz_cluster(temp, data = subset(data_sub, select = -c(year)),
             palette = c("#2E9FDF", "#FF0000", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

##########################################
#NLP, term frequency
##########################################

# stopwords
my_stops <- stopwords(kind = 'en') #c('and', 'of', 'the', 'an', 'of', 'that', 'to', 'in', 'no', 'it', letters)
# clean files
cleaned <- data %>% 
  mutate(tokens = tokenize_words(title, simplify = TRUE, stopwords = my_stops, strip_numeric = TRUE)) %>% 
  dplyr::filter(!is.na(tokens)) %>% 
  dplyr::filter(nchar(tokens) > 1)

# term frequency
nlp <- cleaned %>%
  unnest(cols = 'tokens') %>% 
  group_by(decade, tokens) %>% 
  summarise(n = n()) %>% 
  group_by(decade) %>% 
  mutate(total_words = n()) %>% 
  ungroup() %>% 
  bind_tf_idf(tokens, decade, n) %>% 
  arrange(desc(tf_idf))

# get top 10 for each group
top_10 <-  nlp %>% 
  group_by(decade) %>% 
  slice_max(order_by = tf_idf, n = 10) %>% 
  ungroup()

#Write out NLP data
write.csv(nlp, file = paste0(getwd(), 
                             "/term_freq_results.csv"))
