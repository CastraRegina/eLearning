# edX__HarvardX__Data_Science_Capstone_2019 

Repository for sharing my code samples I created for the
[edx.org](https://www.edx.org)
course
[HarvardX's Data Science Capstone](https://www.edx.org/course/data-science-capstone).  
This course is part of the overall course 
[HarvardX's Data Science Professional Certificate](https://www.edx.org/professional-certificate/harvardx-data-science).



## Content



### Assignment 1: MovieLens Project - Predicting user ratings from MovieLens data set 

Recommendation systems use ratings that users have given items to make specific recommendations to users.
In this project a movie recommendation system will be created, using the features in the MovieLens data set to
predict the potential movie rating a particular user will give for a movie.
A [10M version of the MovieLens data set](http://grouplens.org/datasets/movielens/10m/) is used to provide a training and a validation data set. 

The goal is to train a machine learning algorithm that uses the inputs in one subset to predict movie ratings in the validation set.
For a final test of the algorithm, movie ratings in the validation set have to be predicted as if they were unknown.
RMSE will be used to evaluate how close the predictions are to the true values in the validation set.
A target value of **RMSE <= 0.87750** is given.

Starting with a base model which utilizes just an average rating as a constant value, four different models that take into account movie and/or user effects are used to predict the ratings.
Two of the presented models fulfill the required target value.
The final model reaches a **RMSE** value of approximately **0.8648**.


- **movielens-project-r-script.R**
  [[code](https://github.com/CastraRegina/eLearning/blob/master/edX__HarvardX__Data_Science_Capstone_2019/movielens-project-r-script.R)]
  R file    

- **movielens-project-r-script.Rmd**
  [[code](https://github.com/CastraRegina/eLearning/blob/master/edX__HarvardX__Data_Science_Capstone_2019/movielens-project-report.Rmd)]
  Rmd - R Markdown file    

- **movielens-project-report.pdf**
  [[pdf view github](https://github.com/CastraRegina/eLearning/blob/master/edX__HarvardX__Data_Science_Capstone_2019/movielens-project-report.pdf)]
  [[pdf file](https://github.com/CastraRegina/eLearning/raw/master/edX__HarvardX__Data_Science_Capstone_2019/movielens-project-report.pdf)]\
  Report - pdf file created with *knit* based on the .Rmd file



### Assignment 2: Choose Your Own Project - Time series analysis using the example of weather data 

The programming language **R** is considered to be particularly suitable for the analysis of time series, as there are several
packages and methods available for carrying out predictions. For this project data from a weather station in Jena will be
used to provide training and test data sets.

The goal is to train a machine learning algorithm that uses the daily mean temperatures of four years to predict the 
temperatures of the fifth year. The quality of fitting the signal of the training data set as well as the quality of the forecast of the fifth
year is quantified by a selection of accuracy measurements such as RMSE, MAE, MASE and MAPE.

Starting with a base model which uses just the mean as a constant forecast value, several different models are used.
A neural network method as well as models taking advantage of the seasonality of the data perform quite well in forecasting
the temperatures.


- **CYO-project-r-script.R**
  [[code](https://github.com/CastraRegina/eLearning/blob/master/edX__HarvardX__Data_Science_Capstone_2019/CYO-project-r-script.R)]
  R file    

- **CYO-project-r-script.Rmd**
  [[code](https://github.com/CastraRegina/eLearning/blob/master/edX__HarvardX__Data_Science_Capstone_2019/CYO-project-report.Rmd)]
  Rmd - R Markdown file    

- **CYO-project-report.pdf**
  [[pdf view github](https://github.com/CastraRegina/eLearning/blob/master/edX__HarvardX__Data_Science_Capstone_2019/CYO-project-report.pdf)]
  [[pdf file](https://github.com/CastraRegina/eLearning/raw/master/edX__HarvardX__Data_Science_Capstone_2019/CYO-project-report.pdf)]\
  Report - pdf file created with *knit* based on the .Rmd file



