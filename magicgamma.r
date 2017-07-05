
library(caTools)
library(ggplot2)
library(lattice)
library(caret)
library(class)
library(e1071)
library(rpart)
library(randomForest)
library(GGally)

# Importing the dataset
magicgamma <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"))

head(magicgamma,10)
dim(magicgamma)

#checking for NA's
sum(is.na(magicgamma))

magicgamma <- magicgamma[sample(1:nrow(magicgamma)),]
head(magicgamma,10)

colnames(magicgamma) <- c("fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class")

head(magicgamma,10)

# Encoding the target feature as factor
magicgamma$class = ifelse(magicgamma$class == 'g',1,0)
magicgamma$class = factor(magicgamma$class)
head(magicgamma,10)

ggpairs(magicgamma, columns = 1:10, upper = "blank", aes(colour = class, alpha = 0.8))

# Splitting the dataset into the Training set and Test set
set.seed(123)
split = sample.split(magicgamma$class, SplitRatio = 0.75)
training_set = subset(magicgamma, split == TRUE)
test_set = subset(magicgamma, split == FALSE)

# Fitting Logistic Regression to the Training set
classifier = glm(formula = class ~ .,
                 family = binomial,
                 data = training_set)

summary(classifier)

# Predicting the Test set results
prob_pred <- predict(classifier, type = 'response', newdata = test_set[-11])
y_pred <- ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

# Fitting Logistic Regression to the Training set
classifier = glm(formula = class ~.-fConc-fAsym-fM3Trans-fDist,
                 family = binomial,
                 data = training_set)

summary(classifier)
# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-11])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
# Making the Confusion Matrix
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

#Building KNN classifier
y_pred = knn(train = training_set[,-11],
             test = test_set[, -11],
             cl = training_set[,11],
             k = 5,
             prob = TRUE)

# Predicting the Test set results
# Making the Confusion Matrix
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

# Fitting SVM with a Linear kernel
classifier = svm(formula = class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11])
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

# SVM with a radial kernel
classifier = svm(formula = class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11])
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

#NaiveBayes Classifier for the data
classifier = naiveBayes(x = training_set[-11],
                        y = training_set$class)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11])

confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

# Fitting a Decision Tree classifier
classifier = rpart(formula = class ~ .,
                   data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11], type = 'class')
# Making the Confusion Matrix
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix

# Fitting Random Forests classifier
set.seed(123)
classifier <- randomForest(x = training_set[-11],
                          y = training_set$class,
                          ntree = 500)

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-11])
confusionMatrix(data=y_pred, reference=test_set[,11])
confusionMatrix
