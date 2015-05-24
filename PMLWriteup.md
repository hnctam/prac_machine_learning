# Machine Learning Practical Course Project
Tam Nguyen Chinh Ho  
`r format(Sys.time(), '%d %B, %Y')`  

##Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The data was provided by http://groupware.les.inf.puc-rio.br/har and they allow their data to be used as input for the analysis.

##Getting and Cleaning data


```r
# Libraries declarations
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

Train data and test data are downloaded from URL:

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```
In case the csv files have not yet downloaded to working directory, they will be downloaded and then loaded as inputs to the working.

```r
checkAndDownload <- function(fileName, fileUrl) {
  if (!file.exists(fileName)) {
    #path <- paste("./", fileName)
    download.file(url = fileUrl, destfile = fileName)
  }
}

checkAndDownload("pml-training.csv", trainUrl)
checkAndDownload("pml-testing.csv", testUrl)

# Perform reading csv files, ensure that files are exist

na_values <- c("NA", "#DIV/0!", "")
train_data <- read.csv("pml-training.csv", na.strings = na_values)
test_data <- read.csv("pml-testing.csv"  , na.strings = na_values)
```

In the next step, training data will be partitioned to two parts, train and test data with the ratio as 70%.


```r
set.seed(39916801)

par_indexes <- createDataPartition(y = train_data$classe, p = 0.7, list = FALSE)

par_train_data <- train_data[par_indexes, ]
par_test_data  <- train_data[-par_indexes, ]
```

Now the loaded data needs to be cleaned up to create the better training by removing near zero values and near NaN values which may be noisy to the training.


```r
# Trying to remove the near zero vars
near_zero_vars <- nearZeroVar(par_train_data)

par_train_data <- par_train_data[, -near_zero_vars]
par_test_data <- par_test_data[, -near_zero_vars]

near_na_data <- sapply(par_train_data, function(x) mean(is.na(x))) > 0.95
par_train_data <- par_train_data[, near_na_data==FALSE]
```

Random forests [http://en.wikipedia.org/wiki/Random_forest], the machine learning method, which is selected to run the machine learning.

```r
train_control <- trainControl(method="cv", number=3, verboseIter=FALSE)
fit_model <- train(classe ~ ., data=par_train_data, method="rf", trControl=train_control)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
fit_model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 41
## 
##         OOB estimate of  error rate: 0.01%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3906    0    0    0    0 0.0000000000
## B    1 2657    0    0    0 0.0003762227
## C    0    0 2396    0    0 0.0000000000
## D    0    0    0 2252    0 0.0000000000
## E    0    0    0    0 2525 0.0000000000
```

Now checking out the final model, there are some interesting results which needs to be considered such as the random forest type is classification, number of trees is 500 and number of variables tried at each split is 41.

Next step is trying to predict on the partitioned test data and then checking to see the percentage of accuracy.


```r
predicts <- predict(fit_model, newdata = par_test_data)
confusionMatrix(par_test_data$classe, predicts)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    1 1025    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9998     
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9998     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9991   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   0.9998   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   0.9990   1.0000   1.0000
## Neg Pred Value         1.0000   0.9998   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1937   0.1742   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1742   0.1638   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      1.0000   0.9996   0.9999   1.0000   1.0000
```

Based on the output, the result is quite good with the accuracy is 99.98% and the p-Value is 2.2e-16, so I can come to the conclusion to use this model to predict the real test data.

The final step is to use the random forest model to predict the test data, then output as assignment.


```r
predicts <- predict(fit_model, newdata = test_data)
predicts <- as.character(predicts)

pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

pml_write_files(predicts)
```
