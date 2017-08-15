
# Load the libraries
library(readxl)
library(ggplot2)
library(caTools)

# Read the dataset into the variable credit
credit <- read_excel("default_of_credit_card_clients.xls")

head(credit,10)
dim(credit)

sum(is.na(credit))

summary(credit)

colnames(credit) <- credit[1,]
credit <- credit[-1,]
head(credit)

credit <- sapply(credit, as.numeric)
summary(credit)

#install.packages("ggcorrplot")
library(ggcorrplot)
# Correlation matrix
corr <- round(cor(credit), 1)

# Plot
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 1, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of the Credit Data", 
           ggtheme=theme_bw)

creditm <- credit[,c(-1,-3,-4,-5,-6)]
dim(creditm)
head(creditm)

creditm <- scale(creditm[,-20])

default = credit[,25]
creditdata <- data.frame(creditm, default)
head(creditdata)
dim(creditdata)

#Split into training and test sets
set.seed(123)
split = sample.split(creditdata[,20], SplitRatio = 0.8)
training_set = subset(creditdata, split == TRUE)
test_set = subset(creditdata, split == FALSE)

## Fitting XGBoost to the Training set
#install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[,-20]), label = training_set[,20], nrounds = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[,-20]))
y_pred = ifelse(y_pred >= 0.5,1,0)

# Making the Confusion Matrix
cm = table(test_set[,20], y_pred)
cm

(4418+477)/nrow(test_set)

library(caret)
folds = createFolds(training_set[,20], k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-20]), label = training_set[,20], nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-20]))
  y_pred = ifelse(y_pred >= 0.5,1,0)
  cm = table(test_fold[, 20], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))

accuracy

require(Matrix)
sparse_matrix <- sparse.model.matrix(default~., data = training_set)

classifier <- xgboost(data = sparse_matrix, label = training_set[,20], max.depth = 4,
               eta = 1, nthread = 2, nround = 10,objective = "binary:logistic")

importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = classifier)
head(importance)
