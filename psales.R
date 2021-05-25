
library('readr')
data <- read.csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Decision Tree\\Datasets_DTRF\\Company_Data.csv")
high <- ifelse(data$Sales<5,"no","yes")
data_sal <- cbind(data,high)
data_sal <- data.frame(data_sal)
data_sal <- data_sal[,2:12]
data_sal$high <-as.factor(data_sal$high)
str(data)

library(caTools)
set.seed(0)
split <- sample.split(data$Sales, SplitRatio = 0.8)
data_train <- subset(data, split == TRUE)
data_test <- subset(data, split == FALSE)

library(rpart)
model <- rpart(data_train$Sales ~ ., data = data_train,
               control = rpart.control(cp = 0, maxdepth = 3))

# Plot Decision Tree
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = -3)

# Measure the RMSE on Test data
test_pred <- predict(model, newdata = data_test, type = "vector")

# RMSE
rmse1 <- sqrt(mean(data_test$Sales - test_pred)^2)
rmse1

# Measure the RMSE on Train data
train_pred <- predict(model, newdata = data_train, type = "vector")

# RMSE
rmse_train <- sqrt(mean(data_train$Sales - train_pred)^2)
rmse_train


# Prune the Decision Tree

# Grow the full tree
fullmodel <- rpart(data_train$Sales ~ ., data = data_train,
                   control = rpart.control(cp = 0))

rpart.plot(fullmodel, box.palette = "auto", digits = -3)


# Examine the complexity plot
# Tunning parameter check the value of cp which is giving us minimum cross validation error (xerror)
printcp(fullmodel)   
plotcp(model)

mincp <- model$cptable[which.min(model$cptable[, "xerror"]), "CP"]

# Prune the model based on the optimal cp value
model_pruned_1 <- prune(fullmodel, cp = mincp)
rpart.plot(model_pruned_1, box.palette = "auto", digits = -3)

model_pruned_2 <- prune(fullmodel, cp = 0.02)
rpart.plot(model_pruned_2, box.palette = "auto", digits = -3)

# Measure the RMSE using Full tree
test_pred_fultree <- predict(fullmodel, newdata = data_test, type = "vector")
# RMSE
rmse_f <- sqrt(mean(data_test$Sales - test_pred_fultree)^2)
rmse_f


# Measure the RMSE using Prune tree - model1
test_pred_prune1 <- predict(model_pruned_1, newdata = data_test, type = "vector")
# RMSE
rmse_prune1 <- sqrt(mean(data_test$Sales - test_pred_prune1)^2)
rmse_prune1

# Measure the RMSE using Prune tree - model2
test_pred_prune2 <- predict(model_pruned_2, newdata = data_test, type = "vector")
# RMSE
rmse_prune2 <- sqrt(mean(data_test$Sales - test_pred_prune2)^2)
rmse_prune2

# Prediction for trained data result
train_pred_fultree <- predict(fullmodel, data_train, type = 'vector')

# RMSE on Train Data
train_rmse_fultree <- sqrt(mean(data_train$Sales - train_pred_fultree)^2)
train_rmse_fultree


# Prediction for trained data result
train_pred_prune1 <- predict(model_pruned_1, data_train, type = 'vector')

# RMSE on Train Data
train_rmse_fultree2 <- sqrt(mean(data_train$Sales - train_pred_prune1)^2)
train_rmse_fultree2

# Prediction for trained data result
train_pred_prune2 <- predict(model_pruned_2, data_train, type = 'vector')

# RMSE on Train Data
train_rmse_fultree2 <- sqrt(mean(data_train$Sales - train_pred_prune2)^2)
train_rmse_fultree2


