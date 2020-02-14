
#############################################################################################################
### title:    "HarvardX-PH125.9x: Choose Your Own Project Report"
### subtitle: "Time series analysis using the example of weather data"
#############################################################################################################


#############################################################################################################
### Instructions: (according to "Project Overview: Choose Your Own!")
### For this project, you will be applying machine learning techniques that go beyond standard linear regression. 
### You will have the opportunity to use a publicly available dataset to solve the problem of your choice. 
###    ...
### The ability to clearly communicate the process and insights gained from an analysis is an important skill for data scientists. 
###    ...
### Although the exact format is up to you, the report should include the following at a minimum: 
###   + an introduction/overview/executive summary section that describes the dataset and 
###       summarizes the goal of the project and key steps that were performed; 
###   + a methods/analysis section that explains the process and techniques used, such as data cleaning, data exploration and
###       visualization, any insights gained, and your modeling approach; 
###   + a results section; and 
###   + a conclusion section. 
#############################################################################################################
# Upload remarks:
# As requested 3 files are uploaded:
#   CYO-project-report.pdf
#   CYO-project-report.Rmd
#   CYO-project-r-script.R
#
# Remarks:
#  The pdf file does not contain any source code snippets in order to make it easy to follow.
#  This procedure is common use in academic reports. The source code itself is provided as Rmd & R file.
#  Furthermore for details of the used methods links to references are provided.
#  Dataset is downloaded automatically.
#  All needed libraries will be installed and loaded automatically.
#  Starting from a fresh R installation it might take several runs till all libraries are installed correctly.
#  Using the setup listed in the sessioninfo (see appendix) the provided code runs easily.
#  Running this code takes 15-30 minutes on standard hardware.
#############################################################################################################

print("START:") ; Sys.time() # takes round about 15mins
#############################################################################################################
### Load libraries and install them if not available yet  ... and define standard plot theme
#############################################################################################################
# reference:   https://cran.r-project.org/web/views/TimeSeries.html
# online book:   https://otexts.com/fpp2/
if(!require(colorspace)) install.packages("colorspace")
if(!require(tidyverse))  install.packages("tidyverse")
if(!require(readr))      install.packages("readr")
if(!require(caret))      install.packages("caret")
if(!require(knitr))      install.packages("knitr")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(munsell))    install.packages("munsell")
if(!require(scales))     install.packages("scales")
if(!require(ggplot2))    install.packages("ggplot2")
if(!require(reshape2))   install.packages("reshape2")
if(!require(latex2exp))  install.packages("latex2exp")
if(!require(xts))        install.packages("xts")
if(!require(corrplot))   install.packages("corrplot")
if(!require(ggrepel))    install.packages("ggrepel")
if(!require(grid))       install.packages("grid")
if(!require(forecast))   install.packages("forecast")
if(!require(lubridate))  install.packages("lubridate")
if(!require(quadprog))   install.packages("quadprog")
if(!require(tseries))    install.packages("tseries")
if(!require(zoo))        install.packages("zoo")
if(!require(xts))        install.packages("xts")
if(!require(ggfortify))  install.packages("ggfortify")
if(!require(forecast))   install.packages("forecast")
if(!require(nortest))    install.packages("nortest")


# define consistent ggplot-theme for all plots:
myggtheme <- theme_bw() +
  theme(plot.title   = element_text(size=9, face='bold'),
        axis.title.x = element_text(size=9, face='plain'),
        axis.title.y = element_text(size=9, face='plain'),
        axis.text.x  = element_text(size=8, face='plain'),
        axis.text.y  = element_text(size=8, face='plain'),
        legend.title = element_text(size=8, face='bold'),
        legend.text  = element_text(size=8, face='plain'))
colorsMM<- c(Mean="darkgreen", Median="red")


# define a standard function to create a ggplot of a histogram.
#   input: dataframe = a tibble of data, xData = data for histogram
myhistogram <- function(dataframe, xData, binwidth = 1, title = "", xLabel = "")
{
  median   <- median(xData)
  mean     <- mean(  xData)
  sd       <- sd(    xData)
  annotationText <- paste("Mean = "    , round(mean  , 3), "\n", 
                          "Median = "  , round(median, 3), "\n", 
                          "sigma = "   , round(sd    , 3), "\n",
                          "binwidth = ", binwidth, "\n",
                          "n = ", length(xData), sep="")
  ggplot(dataframe, aes(x = xData)) + 
    geom_histogram(binwidth = binwidth, color="darkblue", fill="lightblue", alpha=0.5) +
    geom_vline(aes(xintercept=median, color="Median"), size=1.5) +
    geom_vline(aes(xintercept=mean  , color="Mean"), size=1.5) +
    scale_colour_manual(name="Colors",values=colorsMM) +
    ggtitle(title) + 
    xlab(xLabel) +
    myggtheme +
    theme(legend.position = c(0.85, 0.85)) +
    annotation_custom(grobTree(textGrob(annotationText, x=0.05, y=0.85, just="left", gp = gpar(fontsize=8) ))) +
    stat_function(fun = function(x) dnorm(x, mean = mean, sd = sd) * length(xData) * binwidth, color = "black", size = 1)
}

set.seed(123)

#############################################################################################################
### Data description / source of data
#############################################################################################################
#
# Weather archive Jena
#   Air temperature, atmospheric pressure, humidity, etc recorded over seven years
#
# ------ Description taken from https://www.kaggle.com/pankrzysiu/weather-archive-jena ----------------------
# Context
#   The Dataset is used by "A temperature-forecasting problem" from the "Deep Learning with Python" book
# Content
#   The data was downloaded from: https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
#   It represents time period between 2009 and 2016
# Acknowledgements
#   The dataset recorded at the Weather Station at the Max Planck Institute for Biogeochemistry in Jena, Germany.
#   https://www.bgc-jena.mpg.de/wetter/
#   It was reassembled by Francois Chollet, the author of the "Deep Learning with Python" book
# Inspiration
#   The main purpose of this dataset is to perform RNN exercise (6.3.1 A temperature-forecasting problem)
#   from the "Deep Learning with Python" book.
# About the file "jena_climate_2009_2016.csv" (12.94 MB)
#   In this dataset, 14 different quantities
#   (such air temperature, atmospheric pressure, humidity, wind direction, and so on)
#   were recorded every 10 minutes, over several years.
#   This example is limited to data from 2009-2016.
# -----------------------------------------------------------------------------------------------------------




#############################################################################################################
### Retrieve / download the data from www.kaggle.com
#############################################################################################################
# Download original file from https://s3.amazonaws.com  (no login credentials needed - very much appreciated for this project) 
#   https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
dl <- tempfile()
download.file("https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip", dl)

# zip file contains one csv-file:
unzip(dl, list=T) %>% knitr::kable() %>% 
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center") %>% 
  row_spec(0, bold = TRUE)



#############################################################################################################
### Data Structure
#############################################################################################################
# csv-file "jena_climate_2009_2016.csv" has following content (first 5 lines):
readLines(unzip(dl, "jena_climate_2009_2016.csv"), n = 5)
# csv-file contains header and is truely comma-separated 

# import csv-file by using the "readr" package
jena_climate_2009_2016_raw <- read_csv(unzip(dl, "jena_climate_2009_2016.csv"))
# view it by using View():
# View(jena_climate_2009_2016_raw)

# Number of columns:
ncol(jena_climate_2009_2016_raw)
# Number of rows (observations):
nrow(jena_climate_2009_2016_raw)

# The 15 attributes have the following names, types and description (taken from the
#   website hosted at kaggle (https://www.kaggle.com/pankrzysiu/weather-archive-jena))
mydata.attributes <- sapply(jena_climate_2009_2016_raw, typeof)
mydata.attributes <- cbind(names(mydata.attributes),
                         mydata.attributes,
                         c("date & time",
                           "atmospheric pressure",
                           "temperature",
                           "potential temperature",
                           "dew point temperature",
                           "relative humidity",
                           "saturation water vapor pressure",
                           "actual water vapor pressure",
                           "water vapor pressure deficit",
                           "specific humidity",
                           "water vapor concentration",
                           "air density",
                           "wind velocity",
                           "maximum wind velocity",
                           "wind direction"))
mydata.attributes <- as_tibble(mydata.attributes, .name_repair = "minimal")
names(mydata.attributes) <- c("Attribute", "Type", "Description")
kable(mydata.attributes) %>% 
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center") %>% 
  row_spec(0, bold = TRUE)

# Show the first 4 rows of data to get a first impression (transposed for better readability):
mydata.first4 <- head(jena_climate_2009_2016_raw, n = 4)
mydata.first4 <- as_tibble(cbind(names(mydata.first4), t(mydata.first4)))
names(mydata.first4) <- c("Attribute", "Observation 1", "Observation 2", "Observation 3", "Observation 4")
kable(mydata.first4) %>% 
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center") %>% 
  row_spec(0, bold = TRUE)




#############################################################################################################
### Data Check and Transformation
#############################################################################################################
# Check for missing values / NA:
mydata.missingValues <- t(colSums(is.na(jena_climate_2009_2016_raw)))
mydata.missingValues
row.names(mydata.missingValues) <- c("Number of NA")
kable(mydata.missingValues) %>%
  row_spec(0, angle = 90) %>% 
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
# --> no missing values.

# Take a look at the statistics of the data:
kable(t(summary(jena_climate_2009_2016_raw))) %>% 
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center", font_size = 9.5) %>% 
  row_spec(0, font_size = 0.1) %>% 
  add_header_above(c("Attribute" = 1, "Statistical Properties" = 6), bold = TRUE)
# --> some attributes show outlier 

# Normalize each attribute to the range of [0,1] and plot it (for plotting: melt it togeher into one Attribute-Value table):
  # mydata.normalized <- melt(as.data.frame(apply(jena_climate_2009_2016_raw[-1], 2, rescale)),
  #                           id.vars = c(), variable.name = "Attribute", value.name = "NormalizedValue")
  # ggplot(mydata.normalized, aes(x = Attribute, y = NormalizedValue)) + geom_boxplot() +
  #   stat_boxplot(geom = "errorbar") + coord_flip() + myggtheme
# --> shows some separated outliers, but does not look well ... try something else ...

# Standardize each attribute (w/o *wv* columns) to mean=0 and 'standard deviation'=1 and plot it
mydata.standardized <- melt(as.data.frame(scale(select(jena_climate_2009_2016_raw, -c(`Date Time`, `max. wv (m/s)`, `wv (m/s)`)))),
                            id.vars = c(), variable.name = "Attribute", value.name = "StandardizedValue")
ggplot(mydata.standardized, aes(x = Attribute, y = StandardizedValue)) + geom_boxplot() +
  stat_boxplot(geom = "errorbar") + coord_flip() + myggtheme
  ylab(TeX("$Standardized\\,Value\\;X' = \\frac{X - \\mu}{\\sigma}$"))
# Ref.: https://stats.stackexchange.com/questions/10289/whats-the-difference-between-normalization-and-standardization
# --> Attributes `rho (g/m**3)` and `p (mbar)` show separated outliers. Remaining attributes seem plausible.

# convert first column "Date Time" into a POSIXct date time format:
jena_climate_2009_2016 <- jena_climate_2009_2016_raw
jena_climate_2009_2016$`Date Time POSIXct` <- as.POSIXct(strptime(jena_climate_2009_2016$`Date Time`, format = "%d.%m.%Y %H:%M:%S", tz="UTC"))
class(jena_climate_2009_2016$`Date Time POSIXct`)

# check periodicity of time series:
periodicity(jena_climate_2009_2016$`Date Time POSIXct`)

# add attribute ID and lag:
jena_climate_2009_2016 <- tibble::rowid_to_column(jena_climate_2009_2016, "ID")
jena_climate_2009_2016 <- jena_climate_2009_2016 %>% mutate(IDname = paste("ID", ID))
lags <- as.numeric(diff(jena_climate_2009_2016$`Date Time POSIXct`, lag = 1))
jena_climate_2009_2016$`lag (min)` <- c(0,lags)

# show the lags != 10min:
jena_climate_2009_2016 %>% select(-c(`Date Time POSIXct`)) %>% filter(`lag (min)` < 10 | 10 < `lag (min)`) %>%
  tail(-1) %>% select(ID, `Date Time`, `lag (min)`) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
# --> there are some observations missing (positive lag values) and others seem to be redundant (negative lag values)

# drop redundant values (78767 <= ID <= 78767+143) (redundant-range1)
jena_climate_2009_2016_clean <- subset(jena_climate_2009_2016, ID < 78767 | 78767+143 < ID)

# drop redundant values (274566 <= ID <= 274566+182) (redundant-range2)
jena_climate_2009_2016_clean <- subset(jena_climate_2009_2016_clean, ID < 274566 | 274566+182 < ID)

# calculate the lags again for the "clean" data:
lags <- as.numeric(diff(jena_climate_2009_2016_clean$`Date Time POSIXct`, lag = 1))
jena_climate_2009_2016_clean$`lag (min)` <- c(0,lags)

# for redundant-range1: visualize the drop-procedure and the resulting data w/o redundancy:
ggplot() +
  geom_line(data = subset(jena_climate_2009_2016_clean, 78767-300 < ID & ID < 78767+300),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "grey" , size=3.0) +
  geom_line(data = subset(jena_climate_2009_2016      , 78767-300 < ID & ID <= 78767    ),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "green", size=1.3) +
  geom_line(data = subset(jena_climate_2009_2016      , 78767     <= ID & ID < 78767+300),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "blue" , size=0.5) +
  xlab("Date Time") + ylab("T (degC)") +
  scale_x_datetime(labels = date_format("%Y-%m-%d\n%H:%M:%S", tz = "UTC")) +
  coord_cartesian(xlim = c(subset(jena_climate_2009_2016_clean,  ID == 78767-160) %>% select(`Date Time POSIXct`) %>% .[[1]],
                           subset(jena_climate_2009_2016_clean,  ID == 78767+160) %>% select(`Date Time POSIXct`) %>% .[[1]]),
                  ylim = c(subset(jena_climate_2009_2016,  78767-144 < ID & ID < 78767) %>% select(`T (degC)`) %>% min(),
                           subset(jena_climate_2009_2016,  78767-144 < ID & ID < 78767) %>% select(`T (degC)`) %>% max())) +
  geom_point(data = subset(jena_climate_2009_2016, ID %in% c(78767-144, 78767, 78766, 78767+143)),
             aes(x = `Date Time POSIXct`, y = `T (degC)`), shape = 21, size = 6, stroke = 1) +
  geom_label_repel(data = subset(jena_climate_2009_2016, ID %in% c(78767-144, 78767, 78766, 78767+143)),
                   aes(x = `Date Time POSIXct`, y = `T (degC)`, label = IDname),
                   box.padding = unit(1.2, 'lines'),
                   point.padding = unit(0.9, 'lines') ) +
  myggtheme 

# for redundant-range2: visualize the drop-procedure and the resulting data w/o redundancy:
ggplot() +
  geom_line(data = subset(jena_climate_2009_2016_clean, 274566-300 < ID & ID < 274566+300),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "grey" , size=3.0) +
  geom_line(data = subset(jena_climate_2009_2016      , 274566-300 < ID & ID <= 274566    ),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "green", size=1.3) +
  geom_line(data = subset(jena_climate_2009_2016      , 274566     <= ID & ID < 274566+300),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "blue" , size=0.5) +
  xlab("Date Time") + ylab("T (degC)") +
  scale_x_datetime(labels = date_format("%Y-%m-%d\n%H:%M:%S", tz = "UTC")) +
  coord_cartesian(xlim = c(subset(jena_climate_2009_2016_clean,  ID == 274566-190) %>% select(`Date Time POSIXct`) %>% .[[1]],
                           subset(jena_climate_2009_2016_clean,  ID == 274566+190) %>% select(`Date Time POSIXct`) %>% .[[1]]),
                  ylim = c(subset(jena_climate_2009_2016,  274566-183 < ID & ID < 274566) %>% select(`T (degC)`) %>% min(),
                           subset(jena_climate_2009_2016,  274566-183 < ID & ID < 274566) %>% select(`T (degC)`) %>% max())) +
  geom_point(data = subset(jena_climate_2009_2016, ID %in% c(274566-183, 274566, 274565, 274566+182)),
             aes(x = `Date Time POSIXct`, y = `T (degC)`), shape = 21, size = 6, stroke = 1) +
  geom_label_repel(data = subset(jena_climate_2009_2016, ID %in% c(274566-183, 274566, 274565, 274566+182)),
                   aes(x = `Date Time POSIXct`, y = `T (degC)`, label = IDname),
                   box.padding = unit(1.2, 'lines'),
                   point.padding = unit(0.9, 'lines') ) +
  myggtheme 
  
# print data w/o redundant-range1:
subset(jena_climate_2009_2016_clean,  78767-3 < ID & ID <  78767+146) %>% select(ID, `Date Time`, `lag (min)`) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# print data w/o redundant-range2:
subset(jena_climate_2009_2016_clean, 274566-3 < ID & ID < 274566+185) %>% select(ID, `Date Time`, `lag (min)`) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# show again the lags != 10min for the data without redundancy:
jena_climate_2009_2016_clean %>% select(-c(`Date Time POSIXct`)) %>% filter(`lag (min)` < 10 | 10 < `lag (min)`) %>%
  tail(-1) %>% select(ID, `Date Time`, `lag (min)`) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
# -> 3 lags are smaller than 60mins. Close these gaps by calculating just the hourly mean():

# aggregate the hourly mean:
jena_climate_2009_2016_aggHourly <- jena_climate_2009_2016_clean %>% 
  select(-c("ID", `Date Time`, "IDname", `lag (min)`)) %>% 
  mutate(`hour` = as.POSIXct(strptime(format(`Date Time POSIXct`, format = "%d.%m.%Y %H"), format = "%d.%m.%Y %H", tz="UTC"))) %>% 
  aggregate(list(`Date Time hour` = .$`hour`), mean) %>%
  mutate(`Date Time` = format(`Date Time hour`, format = "%d.%m.%Y %H:%M:%S")) %>% 
  select(-c(`Date Time POSIXct`, "hour")) %>% 
  mutate(`lag (h)` = c(0,as.numeric(diff(`Date Time hour`, lag = 1)))) %>% 
  tibble::rowid_to_column("IDh")

# show the lags != 1h:
jena_climate_2009_2016_aggHourly %>% select(-c(`Date Time hour`)) %>% tail(-1) %>% filter(`lag (h)` < 1 | 1 < `lag (h)`) %>%
  select(IDh, `Date Time`, `lag (h)`) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# plot raw- and aggregated-data at the place were a lag=30min was detected:
ggplot() +
  geom_line(data = subset(jena_climate_2009_2016_clean, 40379-1 <= ID & ID <= 40379),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "grey" , size=5) +
  geom_line(data = subset(jena_climate_2009_2016_clean, round(40379-0.2*24*6) < ID & ID <= 40379-1),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "black" , size=1) +
  geom_line(data = subset(jena_climate_2009_2016_clean, 40379 <= ID & ID < round(40379+0.2*24*6)),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "black" , size=1) +
  geom_point(data = subset(jena_climate_2009_2016_clean, round(40379-0.2*24*6) < ID & ID < round(40379+0.2*24*6)),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), shape = 21, color = "black" , fill = "white" , size=2.5) +
  xlab("Date Time") + ylab("T (degC)") +
  scale_x_datetime(labels = date_format("%Y-%m-%d\n%H:%M:%S", tz = "UTC")) +
  coord_cartesian(xlim = c(subset(jena_climate_2009_2016_clean,  ID == round(40379-0.1*24*6)) %>% select(`Date Time POSIXct`) %>% .[[1]],
                           subset(jena_climate_2009_2016_clean,  ID == round(40379+0.1*24*6)) %>% select(`Date Time POSIXct`) %>% .[[1]]),
                  ylim = c(subset(jena_climate_2009_2016, round(40379-0.1*24*6) < ID & ID < round(40379+0.1*24*6)) %>% select(`T (degC)`) %>% min(),
                           subset(jena_climate_2009_2016, round(40379-0.1*24*6) < ID & ID < round(40379+0.1*24*6)) %>% select(`T (degC)`) %>% max())) +
  geom_step(data = jena_climate_2009_2016_aggHourly,
            aes(x = `Date Time hour`, y = `T (degC)`), color = "blue", alpha = 0.6, size=1) +
  geom_point(data = subset(jena_climate_2009_2016_clean, ID %in% c(40379-1, 40379)),
             aes(x = `Date Time POSIXct`, y = `T (degC)`), shape = 21, size = 6, stroke = 1) +
  geom_label_repel(data = subset(jena_climate_2009_2016, ID %in% c(40379-1, 40379)),
                   aes(x = `Date Time POSIXct`, y = `T (degC)`, label = IDname),
                   box.padding = unit(0.7, 'lines'),
                   point.padding = unit(0.9, 'lines') ) +
  myggtheme

# plot raw- and aggregated-data for over ca. 3 days:
ggplot() +
  geom_line(data = subset(jena_climate_2009_2016_clean, round(40379-3*24*6) < ID & ID < round(40379+3*24*6)),
            aes(x = `Date Time POSIXct`, y = `T (degC)`), color = "black" , size=0.5) +
  xlab("Date Time") + ylab("T (degC)") +
  scale_x_datetime(labels = date_format("%Y-%m-%d\n%H:%M:%S", tz = "UTC")) +
  coord_cartesian(xlim = c(subset(jena_climate_2009_2016_clean,  ID == round(40379-1.5*24*6)) %>% select(`Date Time POSIXct`) %>% .[[1]],
                           subset(jena_climate_2009_2016_clean,  ID == round(40379+1.6*24*6)) %>% select(`Date Time POSIXct`) %>% .[[1]]),
                  ylim = c(subset(jena_climate_2009_2016, round(40379-1.5*24*6) < ID & ID < round(40379+1.6*24*6)) %>% select(`T (degC)`) %>% min(),
                           subset(jena_climate_2009_2016, round(40379-1.5*24*6) < ID & ID < round(40379+1.6*24*6)) %>% select(`T (degC)`) %>% max())) +
  geom_step(data = jena_climate_2009_2016_aggHourly,
            aes(x = `Date Time hour`, y = `T (degC)`), color = "blue", alpha = 0.6, size=1) +
  myggtheme 

# cut the data off at 24.09.2014 00:00:00 (inclusive)
jena_climate_2009_2014h <- jena_climate_2009_2016_aggHourly %>% 
  subset(`Date Time hour` < as.POSIXct("2014-09-24 00:00:01", tz = "UTC")) %>% 
  select(-c(`lag (h)`))

# clean up:
rm(jena_climate_2009_2016, jena_climate_2009_2016_clean, jena_climate_2009_2016_raw,
   dl, lags, mydata.first4, mydata.missingValues, mydata.standardized,
   jena_climate_2009_2016_aggHourly, mydata.attributes)




#############################################################################################################
### Data Analysis (with respect to time series)
#############################################################################################################
# reduce the clean data to the time range (01.01.2009_00:00:00-31.01.2013_23:00:00) to obtain a dataset with frequency of 365 days (whole year)
jena_climate_2009_2013h <- jena_climate_2009_2014h %>% subset(`Date Time hour` < as.POSIXct("2014-01-01 00:00:00", tz = "UTC"))
head(jena_climate_2009_2013h)
tail(jena_climate_2009_2013h)  

# it turned out during the investigations that the hourly data requires quite a lot of computing resources...
# therefore another aggregation for daily mean data and monthly mean data (based on hourly data) is performed:
# aggregate the daily mean:
jena_climate_2009_2013d <- jena_climate_2009_2013h %>% 
  select(-c("IDh", `Date Time`)) %>% 
  mutate(`day` = as.POSIXct(strptime(format(`Date Time hour`, format = "%d.%m.%Y"), format = "%d.%m.%Y", tz="UTC"))) %>% 
  aggregate(list(`Date Time day` = .$`day`), mean) %>%
  mutate(`Date Time` = format(`Date Time day`, format = "%d.%m.%Y %H:%M:%S")) %>% 
  select(-c(`Date Time hour`, "day")) %>% 
  tibble::rowid_to_column("IDd")
jena_climate_2009_2013d
head(jena_climate_2009_2013d)
tail(jena_climate_2009_2013d)
summary(jena_climate_2009_2013d)

# aggregate the monthly mean:
jena_climate_2009_2013m <- jena_climate_2009_2013h %>% 
  select(-c("IDh", `Date Time`)) %>% 
  mutate(`month` = as.POSIXct(strptime(paste("01.",format(`Date Time hour`, format = "%m.%Y")), format = "%d.%m.%Y", tz="UTC"))) %>% 
  aggregate(list(`Date Time month` = .$`month`), mean) %>%
  mutate(`Date Time` = format(`Date Time month`, format = "%d.%m.%Y %H:%M:%S")) %>% 
  select(-c(`Date Time hour`, "month")) %>% 
  tibble::rowid_to_column("IDm")
jena_climate_2009_2013m
head(jena_climate_2009_2013m)
tail(jena_climate_2009_2013m)
summary(jena_climate_2009_2013m)

# show histogram of hourly, daily, monthly data
myhistogram(jena_climate_2009_2013h, jena_climate_2009_2013h$`T (degC)`, binwidth = 1, xLabel = "T (degC)") +
  ggtitle("Distribution of the hourly \"T (degC)\" in the period 01.01.2009 00:00:00 - 31.01.2013 23:00:00")
myhistogram(jena_climate_2009_2013d, jena_climate_2009_2013d$`T (degC)`, binwidth = 2, xLabel = "T (degC)") +
  ggtitle("Distribution of the daily \"T (degC)\" in the period 01.01.2009 - 31.01.2013")
myhistogram(jena_climate_2009_2013m, jena_climate_2009_2013m$`T (degC)`, binwidth = 3, xLabel = "T (degC)") +
  ggtitle("Distribution of the monthly \"T (degC)\" in the period Jan. 2009 - Dec. 2013")
# --> calculating the mean has a huge influence on distribution, i.e. big difference to normal distribution 

#Shapiro-Wilk-Test normality test
shapiro.test(jena_climate_2009_2013d$`T (degC)`)$p.value
shapiro.test(jena_climate_2009_2013m$`T (degC)`)$p.value
# Anderson-Darling normality test
ad.test(jena_climate_2009_2013h$`T (degC)`)$p.value
ad.test(jena_climate_2009_2013d$`T (degC)`)$p.value
ad.test(jena_climate_2009_2013m$`T (degC)`)$p.value

# put the test results into a nice table for presentation purpose:
normalityTestTb <- tibble('hourly data'  = "N/A", 
                          'daily data'   = format(shapiro.test(jena_climate_2009_2013d$`T (degC)`)$p.value, digits = 3),
                          'monthly data' = format(shapiro.test(jena_climate_2009_2013m$`T (degC)`)$p.value, digits = 3))
normalityTestTb <- add_row(normalityTestTb,  'hourly data'  = format(ad.test(jena_climate_2009_2013h$`T (degC)`)$p.value, digits = 3),
                                             'daily data'   = format(ad.test(jena_climate_2009_2013d$`T (degC)`)$p.value, digits = 3),
                                             'monthly data' = format(ad.test(jena_climate_2009_2013m$`T (degC)`)$p.value, digits = 3))
normalityTestTb <- add_column(normalityTestTb, 'Test on data' = c("Shapiro-Wilk normality test, p-value", "Anderson-Darling normality test, p-value"), .before = 1)
kable(normalityTestTb) %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
rm(normalityTestTb)
# --> the higher the aggregation of the values (=less values overall), the greater the deviation from the normal distribution


# create time-series objects (ts) for the given data (hourly, daily, monthly):
tsJenaTh <- ts(data = jena_climate_2009_2013h$`T (degC)`, start = c(2009,0), frequency = 24*365)
tsJenaTd <- ts(data = jena_climate_2009_2013d$`T (degC)`, start = c(2009,0), frequency = 365)
tsJenaTm <- ts(data = jena_climate_2009_2013m$`T (degC)`, start = c(2009,1), frequency = 12)
tsJenaTh <- window(tsJenaTh, start=c(2009,0.001), end=c(2013,24*365-4)) # because of some reason there are 2008 and 2014-entries!!!
tsJenaTd <- window(tsJenaTd, start=c(2009,0.001), end=c(2013,365)) # because of some reason there are 2008 and 2014-entries!!!

# plot line charts of the 3 datasets:
ggplot()+
  geom_line(data = data.frame(date=time(tsJenaTh)        , TempC=as.matrix(tsJenaTh)), aes(x = date, y = TempC, color = "hourly")) +
  geom_line(data = data.frame(date=time(tsJenaTd)+0.5/365, TempC=as.matrix(tsJenaTd)), aes(x = date, y = TempC, color = "daily")) + 
  geom_line(data = data.frame(date=time(tsJenaTm)+0.5/12 , TempC=as.matrix(tsJenaTm)), aes(x = date, y = TempC, color = "monthly"), size=1) +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Values of \"T (degC)\" in the time period of 01.01.2009 00:00:00 to 01.01.2014 00:00:00 (excl.)") +
  myggtheme +  
  theme(legend.direction = "horizontal", legend.position = c(0.3, 0.05)) +
  scale_color_manual(name = "Data aggregation (mean value)",
                     breaks = c("hourly", "daily", "monthly"),
                     values = c("hourly" = "darkgrey", "daily" = "blue", "monthly" = "black") ) +
  scale_x_continuous()
  
# check the time stamp(s):
time(tsJenaTh)
time(tsJenaTd)
time(tsJenaTm)
cycle(tsJenaTh)
cycle(tsJenaTd)
cycle(tsJenaTm)
deltat(tsJenaTh)
deltat(tsJenaTd)
deltat(tsJenaTm)
frequency(tsJenaTh)
frequency(tsJenaTd)
frequency(tsJenaTm)

# seasonal plot:
suppressMessages(ggseasonplot(tsJenaTh) + myggtheme +
  ggtitle("Seasonal plot of hourly \"T (degC)\" for the years 2009 to 2013") +
    scale_y_continuous(name="T (degC)", limits=c(-25, 35), breaks=c(-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35)) +
    scale_x_continuous(name="Month",
                       breaks = ((1:12)-0.5)/12,
                       labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")))

suppressMessages(ggseasonplot(tsJenaTd) + myggtheme + 
  ggtitle("Seasonal plot of daily \"T (degC)\" for the years 2009 to 2013") +
    scale_y_continuous(name="T (degC)", limits=c(-25, 35), breaks=c(-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35)) +
    scale_x_continuous(name="Month",
                       breaks = ((1:12)-0.5)/12,
                       labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")))

suppressMessages(ggseasonplot(tsJenaTm)+ myggtheme +
  ggtitle("Seasonal plot of monthly \"T (degC)\" for the years 2009 to 2013") +
    scale_y_continuous(name="T (degC)", limits=c(-25, 35), breaks=c(-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35)) +
    scale_x_continuous(name="Month",
                       breaks = ((1:12)-0.5)/13,
                       labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")))
 
# month plot:
suppressMessages(ggmonthplot(tsJenaTh) + myggtheme +
  ggtitle("Month plot of hourly \"T (degC)\" for the years 2009 to 2013") +
    scale_y_continuous(name="T (degC)", limits=c(-25, 35), breaks=c(-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35)) +
    scale_x_continuous(name="Month",
                     breaks = (0:11)*365*2 + 365,
                     labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")))

suppressMessages(ggmonthplot(tsJenaTd) + myggtheme +
  ggtitle("Month plot of daily \"T (degC)\" for the years 2009 to 2013") +
    scale_y_continuous(name="T (degC)", limits=c(-25, 35), breaks=c(-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35)) +
    scale_x_continuous(name="Month",
                       breaks = (0:11)*365/12 + 365*0.5/11,
                       labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")))

suppressMessages(ggmonthplot(tsJenaTm) + myggtheme +
  ggtitle("Month plot of monthly \"T (degC)\" for the years 2009 to 2013") +
    scale_y_continuous(name="T (degC)", limits=c(-25, 35), breaks=c(-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35)) +
    scale_x_continuous(name="Month",
                       breaks = (1:12)+0.5,
                       labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")))

# decompose the time series:
suppressWarnings(autoplot(decompose(tsJenaTh, type = "additive")) + myggtheme + 
                   ggtitle("Decompose plots of hourly \"T (degC)\" for the years 2009 to 2013") )
suppressWarnings(autoplot(decompose(tsJenaTd, type = "additive")) + myggtheme + 
                   ggtitle("Decompose plots of daily \"T (degC)\" for the years 2009 to 2013") )
suppressWarnings(autoplot(decompose(tsJenaTm, type = "additive")) + myggtheme  + 
                   ggtitle("Decompose plots of monthly \"T (degC)\" for the years 2009 to 2013") )
# --> there is seasonality in the signal ; no trend recognizeable 

#autoplot(decompose(tsJenaTd, type = "multiplicative")) + myggtheme  # --> no, does not deliver better results
#autoplot(decompose(tsJenaTm, type = "multiplicative")) + myggtheme  # --> no, does not deliver better results

# check for stationarity: 
### stationarity: Does the data have the same statistical properties throughout the time series? variance, mean, autocorrelation
###    trend stationarity or difference stationarity?  --> test with unit-root test
### augemented Dickey-Fuller test: adf.test() removes autocorrelation and tests for non-stationarity
adf.test(tsJenaTh)$p.value
adf.test(tsJenaTd)$p.value
adf.test(tsJenaTm)$p.value
adf.test(tsJenaTh)$statistic
adf.test(tsJenaTd)$statistic
adf.test(tsJenaTm)$statistic
# --> p-value < 0.05 --> stationarity





#############################################################################################################
#############################################################################################################
# Modelling
#############################################################################################################
#############################################################################################################
# Devide daily data into training and testing dataset: 80% for training (01.01.2009-31.12.2012), holdout of 20% for testing (01.01.2013-31.01.2013)
# Whole years are selected
tsJenaTd.train <- window(tsJenaTd, start=c(2009,1), end=c(2012,365))
tsJenaTd.test  <- window(tsJenaTd, start=c(2013,1), end=c(2013,365))
tsJenaTm.train <- window(tsJenaTm, start=c(2009,1), end=c(2012,12))
tsJenaTm.test  <- window(tsJenaTm, start=c(2013,1), end=c(2013,12))

# Plot train- and test-dataset:
ggplot() +
  forecast::autolayer(tsJenaTd.train, series = 'train data') +
  forecast::autolayer(tsJenaTd.test , series = 'test data') +
  xlab('Year') + ylab('daily mean of \"T (degC)\"') + 
  guides(colour = guide_legend(title = 'Dataset', reverse=T)) +
  myggtheme +
  coord_cartesian(ylim = c(-25,40)) +
  theme(legend.position = c(0.85, 0.1)) +
  scale_color_manual(values=c("green", "black")) 
# --> data shows seasonality
# --> no trend or change in variance recognizeable by eye

# Plot histogram of train dataset "T (degC)" (daily data):
tsJenaTd.train.df <- data.frame(TdegC = as.matrix(tsJenaTd.train), date = time(tsJenaTd.train))
myhistogram(tsJenaTd.train.df, tsJenaTd.train.df$TdegC, binwidth = 2, xLabel = "T (degC)")
# --> has some deviations to a normal distribution 

# Plot histogram of test dataset "T (degC)" (daily data):
tsJenaTd.test.df <- data.frame(TdegC = as.matrix(tsJenaTd.test), date = time(tsJenaTd.test))
myhistogram(tsJenaTd.test.df, tsJenaTd.test.df$TdegC, binwidth = 2, xLabel = "T (degC)")
# --> does not look like a normal distribution and also quite different to the training dataset

# introduce a table for collecting all accuracies of the models:
accuracyTb.train <- tibble() # for collecting all accuracies of the models at the training dataset
accuracyTb.test  <- tibble() # for collecting all accuracies of the models at the test dataset


#############################################################################################################
# Forecasting using simple models 
#############################################################################################################

# -------------------------------------------------------------------------------------------------
# model 1: naive model: project last observation into the future:
#   naive() function from forecast package (snaive for seasonal data)
tsJenaTd.fc_naive  <-  naive(tsJenaTd.train, h=365)
autoplot(tsJenaTd.fc_naive) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 1: Naive method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_naive , tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_naive , tsJenaTd.test)[1,])), Model="M1 naive"         , .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_naive , tsJenaTd.test)[2,])), Model="M1 naive"         , .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_naive$residuals )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 1: naive method"  , "Residuals") + coord_cartesian(xlim = c(-25,20))

# check if normal distribution:
shapiro.test(tsJenaTd.fc_naive$residuals)
# --> not normal!!!

# constant predicted value for test data:
tsJenaTd.fc_naive$mean[1]


# -------------------------------------------------------------------------------------------------
# model 2: seasonal naive model:
#   snaive() for seasonal data forecast month from last year to this year
tsJenaTd.fc_snaive <- snaive(tsJenaTd.train, h=365)
autoplot(tsJenaTd.fc_snaive) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 2: Seasonal naive method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_snaive, tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_snaive, tsJenaTd.test)[1,])), Model="M2 seasonal naive", .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_snaive, tsJenaTd.test)[2,])), Model="M2 seasonal naive", .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_snaive$residuals)))
myhistogram(res.df, res.df$res, 1, "Residuals of model 2: seasonal naive method" , "Residuals") + coord_cartesian(xlim = c(-25,20))

# check if normal distribution:
shapiro.test(tsJenaTd.fc_snaive$residuals)
# --> not normal!!!

# -------------------------------------------------------------------------------------------------
# model 3: average method: calculate mean of data and project that value into the future
#   meanf() function from forecast package 
tsJenaTd.fc_meanf <- meanf(tsJenaTd.train, h=365)
autoplot(tsJenaTd.fc_meanf) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 3: Average method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_meanf , tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_meanf , tsJenaTd.test)[1,])), Model="M3 average"       , .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_meanf , tsJenaTd.test)[2,])), Model="M3 average"       , .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_meanf$residuals )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 3: average method", "Residuals") + coord_cartesian(xlim = c(-25,20))

# check if normal distribution:
shapiro.test(tsJenaTd.fc_meanf$residuals)
# --> not normal!!!

# constant predicted value for test data:
tsJenaTd.fc_meanf$mean[1]

# -------------------------------------------------------------------------------------------------
# model 4: drift method: calculate difference between first and last observation and extrapolate that gradient into the future
#   rwf(, drift=T) function from forecast package
tsJenaTd.fc_drift <- rwf(tsJenaTd.train, h=365, drift=T)
autoplot(tsJenaTd.fc_drift) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 4: Drift method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_drift , tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_drift , tsJenaTd.test)[1,])), Model="M4 drift"         , .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_drift , tsJenaTd.test)[2,])), Model="M4 drift"         , .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_drift$residuals )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 4: drift method"  , "Residuals") + coord_cartesian(xlim = c(-25,20))

# check if normal distribution:
shapiro.test(tsJenaTd.fc_drift$residuals)
# --> not normal!!!

# let us look at the last observation and compare it to the first forecast:
tail(tsJenaTd.fc_drift$x, n=1)
tsJenaTd.fc_drift$mean[1] - (tsJenaTd.fc_drift$mean[2] - tsJenaTd.fc_drift$mean[1])
# --> this method starts at the last observation

# -------------------------------------------------------------------------------------------------
# take a look on the accuracy overview:
accuracyTb.train %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
accuracyTb.test  %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# check for autocorrelation: blue lines = 95% confidence interval
#   acf() : used to identify the moving average (MA) part of the ARIMA model (omit first bar)
acf(tsJenaTd.fc_naive$residuals[2:length(tsJenaTd.fc_naive$residuals)], lag.max = 20, plot=T)
acf(tsJenaTd.fc_snaive$residuals[366:length(tsJenaTd.fc_snaive$residuals)], lag.max = 20, plot=T) # no residuals available for the first year=365days
acf(tsJenaTd.fc_meanf$residuals, lag.max = 20, plot=T)
acf(tsJenaTd.fc_drift$residuals[2:length(tsJenaTd.fc_drift$residuals)], lag.max = 20, plot=T)

#   pacf() : used to identify the autoregressive part (AR)
pacf(tsJenaTd.fc_naive$residuals[2:length(tsJenaTd.fc_naive$residuals)], lag.max = 20, plot=T)
pacf(tsJenaTd.fc_snaive$residuals[366:length(tsJenaTd.fc_snaive$residuals)], lag.max = 20, plot=T) # no residuals available for the first year=365days
pacf(tsJenaTd.fc_meanf$residuals, lag.max = 20, plot=T)
pacf(tsJenaTd.fc_drift$residuals[2:length(tsJenaTd.fc_drift$residuals)], lag.max = 20, plot=T)



#############################################################################################################
# Forecasting using seasonal decomposition + models
#############################################################################################################
# Decompose a time series into seasonal, trend and irregular components using loess, acronym STL.
autoplot(stl(tsJenaTd.train, s.window = "periodic")) + myggtheme

# -------------------------------------------------------------------------------------------------
# model 5: do a forecast using stl + naive:
tsJenaTd.fc_stl_naive   <- stlf(tsJenaTd.train, h=365, method = "naive")
autoplot(tsJenaTd.fc_stl_naive  ) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 5: STL + naive method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_stl_naive  , tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_stl_naive  , tsJenaTd.test)[1,])), Model="M5 STL + naive", .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_stl_naive  , tsJenaTd.test)[2,])), Model="M5 STL + naive", .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_stl_naive$residuals  )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 5: stl + naive method", "Residuals") + coord_cartesian(xlim = c(-25,20))


# -------------------------------------------------------------------------------------------------
# model 6: do a forecast using stl + rwdrift:
tsJenaTd.fc_stl_rwdrift <- stlf(tsJenaTd.train, h=365, method = "rwdrift")
autoplot(tsJenaTd.fc_stl_rwdrift) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 6: STL + drift method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_stl_rwdrift, tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_stl_rwdrift, tsJenaTd.test)[1,])), Model="M6 STL + drift", .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_stl_rwdrift, tsJenaTd.test)[2,])), Model="M6 STL + drift", .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_stl_rwdrift$residuals)))
myhistogram(res.df, res.df$res, 1, "Residuals of model 6: stl + drift method", "Residuals") + coord_cartesian(xlim = c(-25,20))


# -------------------------------------------------------------------------------------------------
# model 7: do a forecast using stl + ARIMA:
tsJenaTd.fc_stl_arima   <- stlf(tsJenaTd.train, h=365, method = "arima")
autoplot(tsJenaTd.fc_stl_arima  ) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 7: STL + ARIMA method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_stl_arima  , tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_stl_arima  , tsJenaTd.test)[1,])), Model="M7 STL + ARIMA", .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_stl_arima  , tsJenaTd.test)[2,])), Model="M7 STL + ARIMA", .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_stl_arima$residuals  )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 7: stl + ARIMA method", "Residuals") + coord_cartesian(xlim = c(-25,20))


# -------------------------------------------------------------------------------------------------
# take a look on the accuracy overview:
accuracyTb.train %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
accuracyTb.test  %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")



#############################################################################################################
# Forecasting using ARIMA
#############################################################################################################
# Autoregressive Integrated Moving Average Modelling univariate time series
# AR - autoregressive part p
# I  - integration, degree of differencing d
# MA - moving average part q
#   integers denoting the grade of order of the three parts
# auto.arima() : determine a good p,d,q automatically

ggtsdisplay(tsJenaTd.train, lag.max = 36)
# arima(p,d,q)
# --> ACF: bars outside 95% confidence? --> Autocorrelation --> MA --> q  
# --> PACF: first bars outside threshold --> AR --> p
# --> no differencing because stationary (no trend visible) --> adf.test()

print("Method auto.arima started: ") ; Sys.time()
# -------------------------------------------------------------------------------------------------
# model 8: seasonal ARIMA. explaination e.g.: https://otexts.com/fpp2/seasonal-arima.html
tsJenaTd.mo_auto.arima1 <- auto.arima(tsJenaTd.train, trace = T, seasonal = T) # ARIMA(5,0,0)(0,1,0)[365]
# better results with: approximation = F & stepwise = F , but very time-consuming!!!
print("Method auto.arima finished: ") ; Sys.time() # ca. 5mins

print("Method forecast of auto.arima started: ") ; Sys.time()
tsJenaTd.fc_auto.arima1 <- forecast(tsJenaTd.mo_auto.arima1, h=365)
print("Method forecast of auto.arima finished: ") ; Sys.time() # <1min

autoplot(tsJenaTd.fc_auto.arima1) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 8: seasonal ARIMA method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracies of forecast model:
accuracy(tsJenaTd.fc_auto.arima1, tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_auto.arima1, tsJenaTd.test)[1,])), Model="M8 seasonal ARIMA", .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_auto.arima1, tsJenaTd.test)[2,])), Model="M8 seasonal ARIMA", .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_auto.arima1$residuals  )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 8: seasonal ARIMA method", "Residuals") + coord_cartesian(xlim = c(-25,20))

# -------------------------------------------------------------------------------------------------
# take a look on the accuracy overview:
accuracyTb.train %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
accuracyTb.test  %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")



#############################################################################################################
# Forecasting using NN
#############################################################################################################
# -------------------------------------------------------------------------------------------------
# model 9: Neural Net method, NNAR(p,P,k)  P=1 1-seasonal-lag (9,1,5)
tsJenaTd.mo_nnetar <- nnetar(tsJenaTd.train)

tsJenaTd.mo_nnetar # NNAR(5,1,4)[365] : 6-4-1 network with 33 weights - linear output units 

print("Method forecast of nnetar started: ") ; Sys.time()
tsJenaTd.fc_nnetar <- forecast(tsJenaTd.mo_nnetar, h=365, PI=T)
print("Method forecast of nnetar finished: ") ; Sys.time() # ca. 8mins

#checkresiduals(tsJenaTd.fc_nnetar)

autoplot(tsJenaTd.fc_nnetar) + myggtheme +
  xlab("Date") +
  ylab("T (degC)") +
  ggtitle("Model 9: Neural Net method, forecast of \"T (degC)\"") +
  coord_cartesian(ylim = c(-25,40))

# show accuracy of forecast model:
accuracy(tsJenaTd.fc_nnetar, tsJenaTd.test) %>%
  kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# gather accuracies of forecast models:
accuracyTb.train <- bind_rows(accuracyTb.train, add_column(as_tibble(t(accuracy(tsJenaTd.fc_nnetar, tsJenaTd.test)[1,])), Model="M9 Neural Net", .before = 1))
accuracyTb.test  <- bind_rows(accuracyTb.test , add_column(as_tibble(t(accuracy(tsJenaTd.fc_nnetar, tsJenaTd.test)[2,])), Model="M9 Neural Net", .before = 1))

# Plot residuals
res.df <- na.omit(data.frame(res = as.matrix(tsJenaTd.fc_nnetar$residuals  )))
myhistogram(res.df, res.df$res, 1, "Residuals of model 9: Neural Net method", "Residuals") + coord_cartesian(xlim = c(-25,20))


# -------------------------------------------------------------------------------------------------
# take a look on the accuracy overview:
accuracyTb.train %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")
accuracyTb.test  %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")



#############################################################################################################
# Results
#############################################################################################################
# define scale function for better visualisation
scale01 <- function(x){((x-min(x))/(max(x)-min(x))*0.88*10 + 0.12*10)*1}

# take a look on the accuracy overview (train dataset):
accuracyTb.train %>% select(Model,RMSE,MAE,MAPE,MASE) %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# scale the train dataset errors and unpivot the table for plotting purpose:
df.accuracyTb01.train <- cbind(select(accuracyTb.train, Model),
                        lapply(select(accuracyTb.train, RMSE,MAE,MAPE,MASE), scale01) ) %>% 
  melt(id = c("Model")) %>% rename(Error = variable)

# plot Errors overview for train dataset:
ggplot(data=df.accuracyTb01.train, aes(x=reorder(Model, desc(Model)), y=value, fill=Error)) +
  geom_bar(stat="identity", position=position_dodge()) +
  scale_y_continuous(trans = 'log10', breaks = c((1:10)), labels = c("small","","","","","","","","","big")) +
  xlab("") +
  ylab("Scaled Error Value") +
  coord_flip() +
  ggtitle("Scaled Error value for train dataset: Which model achieves the lowest Error") +
  myggtheme +
  theme(panel.grid.minor = element_blank())
# --> M7 shows best values

# -------------------------------------------------------------------------------------------------
# take a look on the accuracy overview (test dataset):
accuracyTb.test  %>% select(Model,RMSE,MAE,MAPE,MASE) %>% kable() %>% kable_styling(bootstrap_options = "striped", latex_options = "striped", full_width = FALSE, position = "center")

# scale the test dataset errors and unpivot the table for plotting purpose:
df.accuracyTb01.test <- cbind(select(accuracyTb.test, Model),
                       lapply(select(accuracyTb.test, RMSE,MAE,MAPE,MASE), scale01) ) %>% 
  melt(id = c("Model")) %>% rename(Error = variable)

# plot Errors overview for test dataset:
ggplot(data=df.accuracyTb01.test, aes(x=reorder(Model, desc(Model)), y=value, fill=Error)) +
  #ggplot(data=df.accuracyTb01.test, aes(x=Error, y=value, fill=Model)) +
  geom_bar(stat="identity", position=position_dodge()) +
  #  scale_y_log10(limits = c(0.1,100)) +
  scale_y_continuous(trans = 'log10', breaks = c((1:10)), labels = c("small","","","","","","","","","big")) +
  xlab("") +
  ylab("Scaled Error Value") +
  coord_flip() +
  ggtitle("Scaled Error value for test dataset: Which model achieves the lowest Error") +
  myggtheme +
  theme(panel.grid.minor = element_blank())
# --> M9 shows best values, followed by seasonal methods like M2 & M8


# -------------------------------------------------------------------------------------------------
# plot forecast vs. test data for M7
ggplot() +
  forecast::autolayer(tsJenaTd.train, series = 'train data', size = 1) +
  forecast::autolayer(tsJenaTd.test , series = 'test data' , size = 1) +
  forecast::autolayer(tsJenaTd.fc_stl_arima$fitted, series = 'M7 fitted', size = 0.5) +
  forecast::autolayer(tsJenaTd.fc_stl_arima$mean  , series = 'M7 forecast', size = 0.5) +
  xlab('Year') + ylab('daily mean of \"T (degC)\"') + 
  guides(colour = guide_legend(title = 'Dataset', reverse=F)) +
  myggtheme +
  coord_cartesian(ylim = c(-25,40)) +
  #theme(legend.position = c(0.85, 0.13)) +
  theme(legend.direction = "horizontal", legend.position = c(0.3, 0.05)) +
  scale_color_manual(values=c("cyan", "blue", "green","black"),
                     breaks=c("train data", "test data", "M7 fitted", "M7 forecast"))

# -------------------------------------------------------------------------------------------------
# plot forecast vs. test data for M9
ggplot() +
  forecast::autolayer(tsJenaTd.train, series = 'train data', size = 1) +
  forecast::autolayer(tsJenaTd.test , series = 'test data' , size = 1) +
  forecast::autolayer(tsJenaTd.fc_nnetar$fitted, series = 'M9 fitted'  , size = 0.5) +
  forecast::autolayer(tsJenaTd.fc_nnetar$mean  , series = 'M9 forecast', size = 0.5) +
  xlab('Year') + ylab('daily mean of \"T (degC)\"') + 
  guides(colour = guide_legend(title = 'Dataset', reverse=F)) +
  myggtheme +
  coord_cartesian(ylim = c(-25,40)) +
  theme(legend.position = c(0.85, 0.13)) +
  scale_color_manual(values=c("cyan", "blue", "green","black"),
                     breaks=c("train data", "test data", "M9 fitted", "M9 forecast"))

# -------------------------------------------------------------------------------------------------
# plot forecast vs. test data for M8
ggplot() +
  forecast::autolayer(tsJenaTd.train, series = 'train data', size = 1) +
  forecast::autolayer(tsJenaTd.test , series = 'test data' , size = 1) +
  forecast::autolayer(tsJenaTd.fc_auto.arima1$fitted, series = 'M8 fitted'  , size = 0.5) +
  forecast::autolayer(tsJenaTd.fc_auto.arima1$mean  , series = 'M8 forecast', size = 0.5) +
  xlab('Year') + ylab('daily mean of \"T (degC)\"') + 
  guides(colour = guide_legend(title = 'Dataset', reverse=F)) +
  myggtheme +
  coord_cartesian(ylim = c(-25,40)) +
  theme(legend.position = c(0.85, 0.13)) +
  scale_color_manual(values=c("cyan", "blue", "green","black"),
                     breaks=c("train data", "test data", "M8 fitted", "M8 forecast"))





#############################################################################################################
### Appendix
#############################################################################################################


print("END:") ; Sys.time()

