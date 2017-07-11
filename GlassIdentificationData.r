
library(caTools)
library(ggplot2)
library(lattice)
library(caret)
library(e1071)
library(randomForest)
library(GGally)
library(nnet)

# Importing the dataset
glassdata <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"))

head(glassdata,10)
dim(glassdata)

#checking for NA's
sum(is.na(glassdata))

glassdata <- glassdata[sample(1:nrow(glassdata)),]
head(glassdata,10)

colnames(glassdata) <- c("Idnumber","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","class")

head(glassdata,10)
str(glassdata)

glassdata$class <-factor(glassdata$class)

#Set a reference level using relevel function
glassdata$class <- relevel(glassdata$class, ref ="1")

ggpairs(glassdata, columns = 2:10, upper = "blank", aes(colour = class, alpha = 0.8))

# Splitting the dataset into the Training set and Test set
set.seed(123)
split = sample.split(glassdata$class, SplitRatio = 0.75)
training_set = subset(glassdata, split == TRUE)
test_set = subset(glassdata, split == FALSE)
#dim(training_set)
#dim(test_set)

# Feature Scaling
training_set[,-11] <- scale(training_set[,-11])
test_set[,-11] <- scale(test_set[,-11])

# Fitting Softmax Logistic Regression to the Training set
classifier = multinom(formula = class ~ RI+Na+Mg+Al+Si+K+Ca+Ba+Fe,
                 data = training_set)

summary(classifier)

y_pred <- predict(classifier,newdata = test_set[-11])

confusion_matrix <- table(y_pred, test_set$class)
confusion_matrix

sum(diag(confusion_matrix))/52

# Feature Scaling
training_set[,-11] <- scale(training_set[,-11])
test_set[,-11] <- scale(test_set[,-11])

classifier = svm(formula = class ~ RI+Na+Mg+Al+Si+K+Ca+Ba+Fe,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11])

# Making the Confusion Matrix
cm = table(y_pred, test_set$class)
cm
sum(diag(cm))/52

split = sample.split(glassdata$class, SplitRatio = 0.75)
training_set = subset(glassdata, split == TRUE)
test_set = subset(glassdata, split == FALSE)

library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-11],
                          y = training_set$class,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-11])

# Making the Confusion Matrix
cm = table(y_pred,test_set$class)
cm
sum(diag(cm))/52
