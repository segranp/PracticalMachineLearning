##Practical Machine Learning
##Week 4 
##Prediction Assignment Write-up
#####by Segran Pillay

###Summary

Large amounts personal activity data are collected inexpensively everyday from devices such as Jawbone Up, Nike FuelBand, and Fitbit. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, I utilized data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Each were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

Random Forest and Decision Tree machine learning algorithms, respectively, were applied to 20 test cases available in the [test data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

Key finding was that Random Forest provided a more accurate model fit compared to Decision Tree.


Note: 
- More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) - (see the section on the Weight Lifting Exercise Dataset).
- The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)




###A. Load the data into R


```r
echo = TRUE
#Training data url:
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#Testing data url:
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


# Download Training data:
if (!file.exists("train_data.csv")){
  download.file(train_url, destfile="train_data.csv")
}
# Download Testing data:
if (!file.exists("test_data.csv")){
download.file(test_url, destfile="test_data.csv")
}

# Read Training data and replace missing values & excel division error strings #DIV/0! with 'NA'
train_data <- read.csv("train_data.csv", na.strings=c("NA","#DIV/0!",""), header=TRUE)

# Read Testing data and replace missing values & excel division error strings #DIV/0! with 'NA'
test_data <- read.csv("test_data.csv", na.strings=c("NA","#DIV/0!",""), header=TRUE)

# Summary of Training data classe variable
summary(train_data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


###B. Data Partitioning for cross validation

Training data is split into two data sets by **classe** variable. 
1. Set for training the model 
2. Set for testing the performance the model

Data Split -  60% for training and 40% for testing


```r
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
inTrain <- createDataPartition(y=train_data$classe, p = 0.60, list=FALSE)
training <- train_data[inTrain,]
testing <- train_data[-inTrain,]

dim(training); dim(testing)
```

```
## [1] 11776   160
```

```
## [1] 7846  160
```

###C. Data Processing
####1. Removing metadata - first seven variables omitted to improve model performance


```r
#NearZeroVariance variables removed
training <- training[,-c(1:7)]
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[, nzv$nzv==FALSE]
```

####2. Removing "NA" from training dataset

```r
training_clean <- training
for(i in 1:length(training)) {
  if( sum( is.na( training[, i] ) ) /nrow(training) >= .6) {
    for(j in 1:length(training_clean)) {
      if( length( grep(names(training[i]), names(training_clean)[j]) ) == 1)  {
        training_clean <- training_clean[ , -j]
      }   
    } 
  }
}
training <- training_clean
```

####3. Transforming the Test dataset


```r
columns <- colnames(training)
columns2 <- colnames(training[, -53])
test_data <- test_data[columns2]
dim(test_data)
```

```
## [1] 20 52
```

###D. Cross-Validation
####1. Leveraging **Decision Tree** for Predication

```r
set.seed(1234)
modFit <- rpart(classe ~ ., data=training, method="class")
prediction <- predict(modFit, testing, type="class")
cm <- confusionMatrix(prediction, testing$classe)
print(cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2059  312   38  146   57
##          B   43  718   67   27   58
##          C   71  200 1093  194  175
##          D   28  107   86  816   91
##          E   31  181   84  103 1061
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7325          
##                  95% CI : (0.7225, 0.7422)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6599          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9225  0.47299   0.7990   0.6345   0.7358
## Specificity            0.9015  0.96918   0.9012   0.9524   0.9377
## Pos Pred Value         0.7883  0.78642   0.6307   0.7234   0.7267
## Neg Pred Value         0.9669  0.88461   0.9550   0.9300   0.9403
## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
## Detection Rate         0.2624  0.09151   0.1393   0.1040   0.1352
## Detection Prevalence   0.3329  0.11637   0.2209   0.1438   0.1861
## Balanced Accuracy      0.9120  0.72109   0.8501   0.7935   0.8367
```

**Level of Accuracy:**

```r
overall.accuracy <- round(cm$overall['Accuracy'] * 100, 2)
samplerror <- round(1 - cm$overall['Accuracy'],2)
fancyRpartPlot(modFit)
```

![](Practical_Machine_Learning_Prediction_Assignment_Writeup_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

**Findings:**

 - Obtained 75.35% accuracy on the testing data partitioned from the training
 data
 - The expected out of sample error is roughly 0.25%.
 
 
 

####2. Leveraging **Random Forest** for predication

```r
set.seed(1234)
modFit1 <- randomForest(classe ~ ., data=training)
prediction1 <- predict(modFit1, testing)
cm1 <- confusionMatrix(prediction1, testing$classe)
print(cm1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228   10    0    0    0
##          B    4 1501    6    0    0
##          C    0    7 1362   12    0
##          D    0    0    0 1273    7
##          E    0    0    0    1 1435
## 
## Overall Statistics
##                                          
##                Accuracy : 0.994          
##                  95% CI : (0.992, 0.9956)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9924         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9888   0.9956   0.9899   0.9951
## Specificity            0.9982   0.9984   0.9971   0.9989   0.9998
## Pos Pred Value         0.9955   0.9934   0.9862   0.9945   0.9993
## Neg Pred Value         0.9993   0.9973   0.9991   0.9980   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1913   0.1736   0.1622   0.1829
## Detection Prevalence   0.2852   0.1926   0.1760   0.1631   0.1830
## Balanced Accuracy      0.9982   0.9936   0.9963   0.9944   0.9975
```

**Level of Accuracy:**

```r
overall.accuracy <- round(cm1$overall['Accuracy'] * 100, 2)
samplerror1 <- round(1 - cm1$overall['Accuracy'],2)
plot(modFit1)
```

![](Practical_Machine_Learning_Prediction_Assignment_Writeup_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

**Findings:**
 - Obtained 99.34% accuracy 
 - The expected out of sample error is roughly 0.01%.
 
 
###D. Conclusion
 - Random Forest provides a better model fit compared to Decision Tree.
 - **Course Project Prediction Quiz Portion:**
 
 

```r
 final_prediction <- predict(modFit1, test_data, type="class")
print(final_prediction)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
