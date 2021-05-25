

install.packages("party")
library(party)

library(caret)

library(C50)
install.packages("tree")
library(tree)
library(gmodels)

df <- read.csv(file.choose())

##Exploring and preparing the data ----
str(df)

names(df)[9] <- paste("Class_variable")


df$Class_variable <- as.factor(df$Class_variable)


# Shuffle the data
CD_ <- df[order(runif(768)), ]
str(CD_)

CD_train <- CD_[1:550,]
View(CD_train)
CD_test <- CD_[551:768,]

View(CD_test)


# Step 3: Training a model on the data


credit_model <- C5.0(CD_train[, -9], CD_train$Class_variable)

windows()
plot(credit_model) 

# Display detailed information about the tree
summary(credit_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(credit_model, CD_test[, -9])
test_acc <- mean(CD_test$Class_variable == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(CD_test$Class_variable, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(credit_model, CD_train[, -9])
train_acc <- mean(CD_train$Class_variable == train_res)
train_acc

table(CD_train$Class_variable, train_res)

