
library(party)

library(caret)

library(C50)
library(tree)
library(gmodels)

df <- read.csv(file.choose())

##Exploring and preparing the data ----
str(df)

names(df)[1] <- paste("Position_ofthe_employee")

df$Position_ofthe_employee <- as.factor(df$Position_ofthe_employee)


# Shuffle the data
CD_ <- df[order(runif(196)), ]
str(CD_)

CD_train <- CD_[1:120,]


CD_test <- CD_[121:196,]
credit_model <- C5.0(CD_train[, -1], CD_train$Position_ofthe_employee)

windows()
plot(credit_model) 

# Display detailed information about the tree
summary(credit_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(credit_model, CD_test[, -1])
test_acc <- mean(CD_test$Position_ofthe_employee == test_res)
test_acc

# cross tabulation of predicted versus actual classes
CrossTable(CD_test$Position_ofthe_employee, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(credit_model, CD_train[, -9])
train_acc <- mean(CD_train$Position_ofthe_employee == train_res)
train_acc

table(CD_train$Position_ofthe_employee, train_res)
