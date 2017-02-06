
Practical Machine Learning
--------------------------

Week 4
------

Prediction Assignment Write-up
------------------------------

##### by Segran Pillay

### Summary

Large amounts personal activity data are collected inexpensively everyday from devices such as Jawbone Up, Nike FuelBand, and Fitbit. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, I utilized data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Each were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

Random Forest and Decision Tree machine learning algorithms, respectively, were applied to 20 test cases available in the [test data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

Key finding was that Random Forest provided a more accurate model fit compared to Decision Tree.

Note: - More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) - (see the section on the Weight Lifting Exercise Dataset). - The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

### A. Load the data into R

``` r
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

    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

### B. Data Partitioning for cross validation

Training data is split into two data sets by **classe** variable. 1. Set for training the model 2. Set for testing the performance the model

Data Split - 60% for training and 40% for testing

``` r
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

    ## [1] 11776   160

    ## [1] 7846  160

### C. Data Processing

#### 1. Removing metadata - first seven variables omitted to improve model performance

``` r
#NearZeroVariance variables removed
training <- training[,-c(1:7)]
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[, nzv$nzv==FALSE]
```

#### 2. Removing "NA" from training dataset

``` r
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

#### 3. Transforming the Test dataset

``` r
columns <- colnames(training)
columns2 <- colnames(training[, -53])
test_data <- test_data[columns2]
dim(test_data)
```

    ## [1] 20 52

### D. Cross-Validation

#### 1. Leveraging **Decision Tree** for Predication

``` r
set.seed(1234)
modFit <- rpart(classe ~ ., data=training, method="class")
prediction <- predict(modFit, testing, type="class")
cm <- confusionMatrix(prediction, testing$classe)
print(cm)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1998  321   38  134   56
    ##          B   70  825   70   41   93
    ##          C   51  157 1092  186  177
    ##          D   74  111   86  827   84
    ##          E   39  104   82   98 1032
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7359         
    ##                  95% CI : (0.726, 0.7456)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6645         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8952   0.5435   0.7982   0.6431   0.7157
    ## Specificity            0.9022   0.9567   0.9119   0.9459   0.9496
    ## Pos Pred Value         0.7845   0.7507   0.6566   0.6997   0.7616
    ## Neg Pred Value         0.9558   0.8973   0.9554   0.9311   0.9368
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2547   0.1051   0.1392   0.1054   0.1315
    ## Detection Prevalence   0.3246   0.1401   0.2120   0.1507   0.1727
    ## Balanced Accuracy      0.8987   0.7501   0.8551   0.7945   0.8326

**Level of Accuracy:**

``` r
overall.accuracy <- round(cm$overall['Accuracy'] * 100, 2)
samplerror <- round(1 - cm$overall['Accuracy'],2)
fancyRpartPlot(modFit)
```

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](Practical_Machine_Learning_Prediction_Assignment_Writeup_files/figure-markdown_github/unnamed-chunk-7-1.png)

**Findings:**

-   Obtained 75.35% accuracy on the testing data partitioned from the training data
-   The expected out of sample error is roughly 0.25%.

#### 2. Leveraging **Random Forest** for predication

``` r
set.seed(1234)
modFit1 <- randomForest(classe ~ ., data=training)
prediction1 <- predict(modFit1, testing)
cm1 <- confusionMatrix(prediction1, testing$classe)
print(cm1)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2230   12    0    0    0
    ##          B    1 1500    4    0    0
    ##          C    0    6 1360   13    0
    ##          D    0    0    4 1273   13
    ##          E    1    0    0    0 1429
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9931         
    ##                  95% CI : (0.991, 0.9948)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9913         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9881   0.9942   0.9899   0.9910
    ## Specificity            0.9979   0.9992   0.9971   0.9974   0.9998
    ## Pos Pred Value         0.9946   0.9967   0.9862   0.9868   0.9993
    ## Neg Pred Value         0.9996   0.9972   0.9988   0.9980   0.9980
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2842   0.1912   0.1733   0.1622   0.1821
    ## Detection Prevalence   0.2858   0.1918   0.1758   0.1644   0.1823
    ## Balanced Accuracy      0.9985   0.9937   0.9956   0.9936   0.9954

**Level of Accuracy:**

``` r
overall.accuracy <- round(cm1$overall['Accuracy'] * 100, 2)
samplerror1 <- round(1 - cm1$overall['Accuracy'],2)
plot(modFit1)
```

![](Practical_Machine_Learning_Prediction_Assignment_Writeup_files/figure-markdown_github/unnamed-chunk-9-1.png)

**Findings:** - Obtained 99.34% accuracy - The expected out of sample error is roughly 0.01%.

### D. Conclusion

-   Random Forest provides a better model fit compared to Decision Tree.
-   **Course Project Prediction Quiz Portion:**

``` r
 final_prediction <- predict(modFit1, test_data, type="class")
print(final_prediction)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
