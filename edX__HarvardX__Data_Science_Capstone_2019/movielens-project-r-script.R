

###################################################################################################
###################################################################################################
## Retrieve the data as described by the project instructions 
###################################################################################################
###################################################################################################


#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


###################################################################################################
###################################################################################################
# Install some packages and load them...
###################################################################################################
###################################################################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra")


###################################################################################################
###################################################################################################
### Check and explore the data
###################################################################################################
###################################################################################################

# What does the data look like? --> Show number of rows and columns
str(edx)


# Each row represents a rating given by one user to one movie:
head(edx)


# How similar are the train and test dataset? --> in summary very similar
summary(edx)
summary(validation)


# Are there any missing values? --> all zero (=no-N/A), that's good 
colSums(is.na(edx)) 
colSums(is.na(validation))
anyNA(edx)
anyNA(validation)


# How many unique users provided ratings and for how many unique movies:
edx %>% 
  summarize(n_userIds  = n_distinct(userId),
            n_movieIds = n_distinct(movieId),
            n_titles   = n_distinct(title),
            n_genres   = n_distinct(genres)) %>% knitr::kable()

validation %>% 
  summarize(n_userIds  = n_distinct(userId),
            n_movieIds = n_distinct(movieId),
            n_titles   = n_distinct(title),
            n_genres   = n_distinct(genres))%>% knitr::kable()

# In both datasets the number of "movieId" and "title" is not the same:
#   there is one "title" with two "movieId"s. Which movie-"title"?  
edx %>% group_by(title,movieId) %>%
  summarize(count_title = n()) %>%
  group_by(title) %>% 
  summarize(count_movieId = n()) %>%  
  arrange(desc(count_movieId)) %>%
  slice(1)

# Movie "War of the Worlds (2005)" has two different "movieIds":
edx %>% filter(title == "War of the Worlds (2005)") %>% 
  group_by(movieId, title, genres) %>% 
  summarize()

# Same issue in "validation" dataset:
#   Movie "War of the Worlds (2005)" has two different "movieIds":
validation %>% filter(title == "War of the Worlds (2005)") %>% 
  group_by(movieId, title, genres) %>% 
  summarize()
# --> Usually I would unite these two entries into one unique entry.
#     But I think we are not allowed to mess around with the provided data.
#     In the following we will identify a movie by "movieId". 
#     The "title" is just for information.
#     So from here on we think of having two different movies using ids 34048 and 64997...



# Plotting the distribution of rating for training set (edx):
colorsMM<- c(Mean="darkgreen", Median="red")
p1 <- ggplot() + 
  geom_histogram(aes(edx$rating), binwidth=0.25, color="darkblue", fill="lightblue", alpha=0.5) +
  geom_vline(aes(xintercept=median(edx$rating), color="Median"), size=2) +
  geom_vline(aes(xintercept=mean(edx$rating), color="Mean"), size=2) +
  scale_colour_manual(name="Colors",values=colorsMM) +
  ggtitle("Distribution of \"rating\" for the edx dataset") +
  xlab("rating") +
  theme_bw() +
  theme(legend.position = c(0.15, 0.90))
# Plotting the distribution of rating for testing set (validation):
p2 <- ggplot() + 
  geom_histogram(aes(validation$rating), binwidth=0.25, color="orangered", fill="orange", alpha=0.5) +
  geom_vline(aes(xintercept=median(validation$rating), color="Median"), size=2) +
  geom_vline(aes(xintercept=mean(validation$rating), color="Mean"), size=2) +
  scale_colour_manual(name="Colors",values=colorsMM) +
  ggtitle("Distribution of \"rating\" for the validation dataset") + 
  xlab("rating") +
  theme_bw() +
  theme(legend.position = c(0.15, 0.90))
grid.arrange(p1, p2, ncol=2)
# --> similar distributions
# --> the half-number-ratings are less common than the whole-number-ratings

# The distributions show there are 10 distinct values of "rating",
#   beginning with 0.5 in steps of 0.5 till max rating of 5 (incl.):
nRatings_edx        <- edx        %>% group_by(rating) %>% arrange(rating) %>% summarize(n_edx = n())
nRatings_validation <- validation %>% group_by(rating) %>% arrange(rating) %>% summarize(n_validation = n())
merge(x = nRatings_edx, y = nRatings_validation, by = "rating", all = TRUE) %>%
  mutate("% of n_edx"        = round(n_edx       /sum(n_edx)       *100,1)) %>% 
  mutate("% of n_validation" = round(n_validation/sum(n_validation)*100,1)) %>% knitr::kable()



# Some movies get rated more than others:
edx %>% 
  count(movieId) %>%
  arrange(desc(n)) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, color="black", fill="lightgrey") +
  geom_vline(aes(xintercept=median(n), color="Median"), size=2) +
  geom_vline(aes(xintercept=mean(n), color="Mean"), size=2) +
  scale_colour_manual(name="Colors",values=colorsMM) +
  scale_x_log10() +
  ggtitle("Movies") +
  xlab("Number of ratings n") +
  theme_bw() +
  theme(legend.position = c(0.90, 0.90))

# What is the maximum number of ratings a movie got?
edx %>% 
  group_by(movieId, title) %>%
  summarize(n = n()) %>% 
  arrange(desc(n)) %>%
  head(8) %>% 
  knitr::kable()
# --> "Pulp Fiction (1994)" has the greatest number of ratings: 31362

# Is there some correlation between average rating and number of ratings with respect of movies 
p <- edx %>% 
  group_by(movieId) %>% 
  summarize(avg_rating=mean(rating), n_movieId = n()) %>% 
  arrange(desc(avg_rating)) %>% 
  ggplot(aes(x=n_movieId, y=avg_rating)) +
    geom_point(shape=1) +
    geom_smooth(se=FALSE, size=2) + 
    scale_x_log10() +
    ggtitle("Movies") +
    xlab("Number of ratings n") +
    ylab("Average of ratings") +
    theme_bw()
suppressMessages(print(p))
# --> Movies with a high number of ratings tend to have higher ratings 



# Some users are more active than others at rating movies
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, color="black", fill="lightgrey") + 
  geom_vline(aes(xintercept=median(n), color="Median"), size=2) +
  geom_vline(aes(xintercept=mean(n), color="Mean"), size=2) +
  scale_colour_manual(name="Colors",values=colorsMM) +
  scale_x_log10() + 
  ggtitle("Users") +
  xlab("Number of ratings n") +
  theme_bw() +
  theme(legend.position = c(0.90, 0.90))

# How many times did the most active user rate a movie?
edx %>% 
  count(userId) %>% 
  arrange(desc(n)) %>% slice(1)
# --> The most active user rated up to 6616 movies

# Take a look at the average ratings of the users:
edx %>% 
  group_by(userId) %>% 
  summarize(user_avg_rating=mean(rating)) %>%  
  ggplot() + 
  geom_histogram(aes(user_avg_rating), bins = 30, color="black", fill="grey", alpha=0.5)+
  geom_vline(aes(xintercept=median(user_avg_rating), color="Median"), size=2) +
  geom_vline(aes(xintercept=mean(user_avg_rating), color="Mean"), size=2) +
  scale_colour_manual(name="Colors",values=colorsMM) +
  ggtitle("Users") +
  xlab("User's average rating") +
  theme_bw() +
  theme(legend.position = c(0.10, 0.90))
# --> there is substantial variability across users

# Is there some correlation between average rating and number of ratings with respect of users
p <- edx %>% 
  group_by(userId) %>% 
  summarize(avg_rating=mean(rating), n_userId = n()) %>% 
  arrange(desc(avg_rating)) %>% 
  ggplot(aes(x=n_userId, y=avg_rating)) +
  geom_point(shape=1) +
  geom_smooth(se=FALSE, size=2) + 
  scale_x_log10() +
  ggtitle("Users") +
  xlab("Number of ratings n") +
  ylab("Average of ratings") +
  theme_bw()
suppressMessages(print(p))
# --> hmmm, does not look to be significant 

# clean up:
rm(nRatings_edx, nRatings_validation, p1, p2, p)


###################################################################################################
###################################################################################################
### Prediction modelling 
###################################################################################################
###################################################################################################

# Define a loss-function that computes the Residual Mean Squared Error ("typical error").
#   By using this function several models will be assessed. 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


###################################################################################################
# Model 0: Use the average of ratings as a baseline model 
###################################################################################################
mu_hat <- mean(edx$rating)
mu_hat

# validate the model:
rmse_m0 <- RMSE(validation$rating, mu_hat)

# store the result in a table to compare it with other models:
rmse_results <- tibble(Method = "M0: Just the average", RMSE = rmse_m0)
rmse_results %>% knitr::kable()

# clean up:
rm(rmse_m0, mu_hat)



###################################################################################################
# Model 1: Movie Effect Model
###################################################################################################
mu <- mean(edx$rating) 

# Introduce b_i to represent average ranking for movie i, i.e. difference to overall average mu:
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# plot the distribution of b_i:
movie_avgs %>% ggplot() + 
  geom_histogram(aes(b_i), binwidth=0.5, color="black", fill="grey", alpha=0.5) +
  xlab("b_i") +
  theme_bw()

# create predictions for test-set:
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

# validate the model:
rmse_m1 <- RMSE(validation$rating, predicted_ratings)

# store the result in a table to compare it with other models:
rmse_results <- bind_rows(rmse_results, tibble(Method="M1: Movie Effect Model", RMSE = rmse_m1 ))
rmse_results %>% knitr::kable()

# clean up:
rm(rmse_m1, predicted_ratings, movie_avgs, mu)



###################################################################################################
# Model 2: Movie + User Effects Model
###################################################################################################
# Looking at the users' average ratings, a substantial variability across users is noticeable.
#   Let's take this into account.
mu <- mean(edx$rating) 

# Introduce b_i to represent average ranking for movie i, i.e. difference to overall average mu:
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Introduce b_u to represent a user-specific effect
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# create predictions for test-set:
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs,  by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# validate the model:
rmse_m2 <- RMSE(validation$rating, predicted_ratings)

# store the result in a table to compare it with other models:
rmse_results <- bind_rows(rmse_results, tibble(Method="M2: Movie + User Effects Model", RMSE = rmse_m2 ))
rmse_results %>% knitr::kable()

# clean up:
rm(rmse_m2, predicted_ratings, movie_avgs, user_avgs, mu)



###################################################################################################
# Model 3: Regularized Movie Effect Model
###################################################################################################
# Large estimates that come from small sample sizes will be penalized.
#   Lambda is introduced as a penalty factor. 
#   Lambda is a tuning parameter. It will be chosen by cross-validation.

# Implement Model 3: Regularized Movie Effect Model
rmse_modelM3 <- function(lambda, train_set, test_set){
  mu <- mean(train_set$rating)
  
  just_the_sum <- train_set %>%
    group_by(movieId) %>%
    summarize(s = sum(rating - mu), n_i = n())

  predicted_ratings <- test_set %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+lambda)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred

  RMSE(test_set$rating, predicted_ratings)  
}

# For cross-validation the dataset edx has to be devided into train_set and test_set: 
set.seed(755)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set  <- edx[-test_index,]
test_set   <- edx[ test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Iterate lambda using following range&steps:
lambdas <- seq(0, 5.75, 0.25)

# validate the model for several lambdas
rmses <- sapply(lambdas, function(lambda){
  rmse_modelM3(lambda, train_set, test_set)
})

# plot the results:
tibble(lambdas=lambdas, rmses=rmses) %>% 
  ggplot(aes(x=lambdas, y=rmses)) +
    geom_point(shape=19) +
    xlab("lambda") +
    ylab("RMSE") +
    ggtitle("Cross validation on dataset edx") +
    theme_bw()

# retrieve the best lambda:
lambda <- lambdas[which.min(rmses)]

mu <- mean(edx$rating) 

# For plotting calculate the the tibbles movie_avg with original b_i and movie_reg_avgs with regularized b_i.
# Introduce b_i to represent average ranking for movie i, i.e. difference to overall average mu:
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Here the regularized b_i are stored:
movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

# plot of the regularized estimates versus the original estimates.
tibble(original = movie_avgs$b_i,
       regularlized = movie_reg_avgs$b_i,
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) +
  geom_point(shape=1, alpha=0.5) +
  xlab("original b_i") +
  ylab("regularized b_i") +
  theme_bw()

# validate the model:
rmse_m3 <- rmse_modelM3(lambda, edx, validation)

# store the result in a table to compare it with other models:
rmse_results <- bind_rows(rmse_results, tibble(Method="M3: Regularized Movie Effect Model", RMSE = rmse_m3 ))
rmse_results %>% knitr::kable()

# clean up:
rm(rmse_m3, movie_avgs, movie_reg_avgs, lambda, lambdas, rmses, mu, test_index, test_set, train_set, rmse_modelM3)



###################################################################################################
# Model 4: Regularized Movie + User Effect Model
###################################################################################################
# Lambda is introduced as a penalty factor regarding movie and user effects. 
# Lambda is a tuning parameter. It will be chosen by cross-validation.

# Implement Model 4: Regularized Movie + User Effect Model:
rmse_modelM4 <- function(lambda, train_set, test_set){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(test_set$rating, predicted_ratings))
}

# For cross-validation the dataset edx has to be devided into train_set and test_set: 
set.seed(755)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set  <- edx[-test_index,]
test_set   <- edx[ test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Iterate lambda using following range&steps:
lambdas <- seq(0, 12, 0.25)

# validate the model for several lambdas
rmses <- sapply(lambdas, function(lambda){
  rmse_modelM4(lambda, train_set, test_set)
})

# plot the results:
tibble(lambdas=lambdas, rmses=rmses) %>% 
  ggplot(aes(x=lambdas, y=rmses)) +
  geom_point(shape=19) +
  xlab("lambda") +
  ylab("RMSE") +
  ggtitle("Cross validation on dataset edx") +
  theme_bw()

# retrieve the best lambda:
lambda <- lambdas[which.min(rmses)]

# validate the model:
rmse_m4 <- rmse_modelM4(lambda, edx, validation)

# store the result in a table to compare it with other models:
rmse_results <- bind_rows(rmse_results, tibble(Method="M4: Regularized Movie + User Effect Model", RMSE = rmse_m4 ))
rmse_results %>% knitr::kable()

# clean up:
rm(rmse_m4, lambda, lambdas, rmses, test_index, test_set, train_set, rmse_modelM4)



###################################################################################################
# Summary
###################################################################################################
# Show all results:
rmse_results %>% knitr::kable()
# |Method                                    |      RMSE|
# |:-----------------------------------------|---------:|
# |M0: Just the average                      | 1.0612018|
# |M1: Movie Effect Model                    | 0.9439087|
# |M2: Movie + User Effects Model            | 0.8653488| <----
# |M3: Regularized Movie Effect Model        | 0.9438528|
# |M4: Regularized Movie + User Effect Model | 0.8648201| <----

# Methods M2 and M4 deliver RMSE values lower than the required value of 0.87750:
rmse_results %>% filter(RMSE < 0.87750) %>% knitr::kable()
# |Method                                    |      RMSE|
# |:-----------------------------------------|---------:|
# |M2: Movie + User Effects Model            | 0.8653488|
# |M4: Regularized Movie + User Effect Model | 0.8648201|

# Best method is:
bestRMSE <- rmse_results[which.min(rmse_results$RMSE),] 
# its name is:
bestRMSE$Method
# ... and it achieves a RMSE value of about 0.8648:
round(bestRMSE$RMSE, digits =4)

