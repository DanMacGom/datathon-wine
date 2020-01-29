# ----------------------------------

# EXPLANATION 

# This dataset deals with our favourite beverage: wine.
# It is about a white variant of the Portuguese "Vinho Verde".
# Vinho verde is a unique product from the Minho (northwest) region of Portugal. 
# Medium in alcohol, it is particularly appreciated due to its freshness (specially in the summer).

# The goal is to model wine quality based on physicochemical tests and sensory data.
# There is no data about grape types, brand, selling price, etc.
# As the class is ordered, you could choose to do a regression instead of a classification.

# The prediction will have to be uploaded in one of your public repository in your github

# Deadline is at 5.30pm!

# For every minute of late delivery your final accuracy will be decreased by 0.05
# For example If your final accuracy is 86% but you deliver 20 minutes late then
# your final accuracy will be 0.86 - (0.05 x 20 = 1) = 0.85
#
# No extra point for early delivery.

# Let's start !

# ----------------------------------

# Import Libraries
library(rstudioapi)
library(ggplot2)
library(caret)

# Load the data
setwd(dirname(getActiveDocumentContext()$path))
df <- read.csv("data/training.csv", sep = ",", header = TRUE)

# See row and columns numbers
dim(df)

# Look at the classes of the features
str(df)

# Look at the name of the features
names(df)

# Look for missing values
sum(is.na(df))

# Change the dependent variable as factor
df$quality <- as.factor(df$quality)

# Look at the distribution of the dependent variable
ggplot(df, aes(quality)) +
  geom_bar()  +
  labs(title=" Dependent variable distribution",
        x ="Quality of the wine",
        y = "")
  
# Run model
t.control <- trainControl(method = "cv", number = 10)
model <- train(quality ~ ., data = df, methode = "rf", trcontrol = t.control)

model$results # ~ 65% accuracy

# Once you have your predicsion, load the validation and fill the quality feature
validation <- read.csv("data/validation.csv", sep = ",", header = TRUE)

print(validation$quality)

# Good luck