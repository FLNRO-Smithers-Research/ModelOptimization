#############Machine Learning evaluation scripts
#various code chunks designed to evaluate and build machine learning
##Will MacKenzie March 2019

.libPaths("E:/R packages351")
#.libPaths("F:/R/Packages")
require (RGtk2)
require(plyr)
require (rChoiceDialogs)
require (data.table)
require(doBy)
require (utils)
require(labdsv)
require(tools )
require(svDialogs)
require(tcltk)
require(foreach)
require(dplyr)
require(reshape2)
require(reshape)
require(doParallel)
require(dostats)
require (stringr)

########Packages of models to be tested
require(randomForest)
require(C50)
require(ada)

###Packages for model optimization
require(UBL)  ####for balancing training points
require(caret)##### for evaluating models
require (pROC)
require (caretEnsemble)

##Set up workspace
rm(list=ls())
wd=tk_choose.dir(); setwd(wd)
options(stringsAsFactors = FALSE)
###OPTIONAL: Set up to run loops in parallel###
require(doParallel)
set.seed(123321)
coreNum <- as.numeric(detectCores()-1)
coreNo <- makeCluster(coreNum)
registerDoParallel(coreNo, cores = coreNum)
clusterEvalQ(coreNo, .libPaths("E:/R packages351"))


######Load Training Data
#########Always name table as 'X1' for the tests
fplot=(file.choose()) 
X1 <- fread(fplot, stringsAsFactors = FALSE, data.table = FALSE)#
 
####change the name of the categorical or response variable to be modelled
str (X1) ###examine to find variable
colnames(X1)[4]=c("Class") ## choose column which is the response variable and rename it
colnames(X1)[2]=c("Group") ## choose column which is the grouping variable and rename it
X1$Class <- as.factor(X1$Class)
#########this block of info can be inserted whereever required in the script to split off information/blocks 
str (X1) ###examine to find information of blocking variables
ID_col <- c(1, 2, 3) ###columns which are identifiers and not variables
X1_ID <- X1 [, c(ID_col)] ###ID data 
X1.sub <- X1 [, -c(ID_col)] ###Data to be tested

prop.table(table(X1.sub$Class))



# calculate correlation matrix of variables
correlationMatrix <- cor(X1.sub[,-c(1)])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9, verbose = TRUE, names = TRUE) ## review for removal of highly correlated vars


########Training Point balancing of uncommon classes UBL package
X1.sub2 <- SmoteClassif(Class ~ ., X1.sub, C.perc = "balance", k= 5 , repl = FALSE, dist = "Euclidean") # creates full balanced data set
X1.sub2 <- SmoteClassif(Class ~ ., X1.sub, C.perc = list(E4 = 50, E5 = 100), k= 5 , repl = FALSE, dist = "Euclidean") # creates manusl balancing - C.perc list is variables and weights
prop.table(table(X1.sub2$Class)) ## check percentage of each 
X1.sub <- X1.sub2
########Random Over/Under samplin of uncommon classes UBL package - LESS USEFUL
#X1.sub <- RandUnderClassif(ESuit ~ ., X1.sub)
#X1.sub <- downSample(x= X1.sub[-1], y= X1.sub$ESuit)


#####test for mtry in rF - can be time consuming
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid", classProbs = TRUE)
set.seed(123456)
tunegrid <- expand.grid(.mtry=c(1:10))
X1.sub$Class <- as.factor(X1.sub$Class)
droplevels(X1.sub)
#X1.sub$ESuit <- make.names(X1.sub$ESuit, unique = FALSE, allow_ = TRUE)
rf_gridsearch <- train(Class ~ ., data=X1.sub, method='rf', metric= 'Accuracy', tuneGrid=tunegrid,
                       trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)


###########a caret test for training set size using a Learing Curve
set.seed(29510)
X1.sub <- na.omit(X1.sub)
X1.sub$Class <- as.factor(X1.sub$Class)
droplevels(X1.sub)
lda_data <- learing_curve_dat(dat = X1.sub,
                              proportion = (1:10)/10,
                              outcome = "Class",
                              test_prop = 1/5,
                              ## `train` arguments:
                              method = "rf",
                              metric = "Accuracy", na.omit = TRUE,
                              trControl = trainControl(classProbs = FALSE, method = "cv",
                                                       verbose = TRUE))#, summaryFunction = twoClassSummary))

ggplot(lda_data, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw()

######A CARET test for recursive feature selection

####for testing
X2 <- X1
X1 <- X2
group <- "Pl" ###select a group and reduce dataset to that group
X1.sub <- X1.sub[(X1$Group %in% group),]
X1.sub$Class <- factor(X1.sub$Class, levels=unique(X1.sub$Class))


control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = TRUE)
#outcomeName <- 'Class'
#outcomeName <- as.factor (outcomeName)
X1.sub$Class <- as.factor(X1.sub$Class)
droplevels(X1.sub)
predictors <- names(X1.sub)[!names(X1.sub) %in% outcomeName]
VariableTest <- rfe(X1.sub[,predictors], X1.sub[,c("Class")],
                    rfeControl = control)
VariableTest
predictors(VariableTest) #top variables
plot (VariableTest, type = c("o", "g")) ###plots the increase in model performance by number of variables

############

#######################Compare Models
#https://rdrr.io/cran/caret/man/models.html list available models for Caret
names(getModelInfo()) ###returns names of all possible caret models
methodList = c("rf", "ada", "C50") # create a list of caret models to be run
set.seed (12345)
control <- trainControl(method="cv", number=5, returnResamp = "final",
                        classProbs = TRUE, 
                        search = "random",
                        repeats=3)#

tune.grid <- expand.grid(.cp=0)
# C5.0
droplevels(X1.sub$Class)
###c50 modle
fit.c50 <- train(Class ~., X1.sub, method="C5.0", metric= "Accuracy", trControl=control)#[,c(1:4)]
#Random Forest Model
fit.rf <- train(Class ~.,  X1.sub, method='rf', metric="Accuracy", trControl=control, verbose=FALSE)
#rpart
fit.tb <- train(Class ~.,  X1.sub, method='treebag', metric="Accuracy", trControl=control, verbose=FALSE)
# compare models
modelcompare <- resamples(list(c5.0=fit.c50, rf=fit.rf, tb=fit.tb))
modelcompare
summary(modelcompare)
dotplot(modelcompare)

# Example of Boosting Algorithms
#https://topepo.github.io/caret/model-training-and-tuning.html#grids
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

fit.gbm <- train(Class ~.,  X1.sub, method='gbm', metric="Accuracy", trControl=control, verbose=FALSE, tuneGrid = gbmGrid )
trellis.par.set(caretTheme())
plot(fit.gbm) 
plot(fit.gbm, metric = "Kappa")

####Modelling Ensembles with caret

###Full source code (also on GitHub): https://amunategui.github.io/blending-models/#sourcecode
#### see also https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html
  
  library(caret)
names(getModelInfo())


# shuffle and split the data into three parts
set.seed(1234)
vehicles <- vehicles[sample(nrow(vehicles)),]
split <- floor(nrow(vehicles)/3)
ensembleData <- vehicles[0:split,]
blenderData <- vehicles[(split+1):(split*2),]
testingData <- vehicles[(split*2+1):nrow(vehicles),]

# set label name and predictors
labelName <- 'cylinders'
predictors <- names(ensembleData)[names(ensembleData) != labelName]

library(caret)
# create a caret control object to control the number of cross-validations performed
myControl <- trainControl(method='cv', number=3, returnResamp='none')

# quick benchmark model 
test_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)
preds <- predict(object=test_model, testingData[,predictors])

library(pROC)
auc <- roc(testingData[,labelName], preds)
print(auc$auc) # Area under the curve: 0.9896

# train all the ensemble models with ensembleData
model_gbm <- train(ensembleData[,predictors], ensembleData[,labelName], method='gbm', trControl=myControl)
model_rpart <- train(ensembleData[,predictors], ensembleData[,labelName], method='rpart', trControl=myControl)
model_treebag <- train(ensembleData[,predictors], ensembleData[,labelName], method='treebag', trControl=myControl)

# get predictions for each ensemble model for two last data sets
# and add them back to themselves
blenderData$gbm_PROB <- predict(object=model_gbm, blenderData[,predictors])
blenderData$rf_PROB <- predict(object=model_rpart, blenderData[,predictors])
blenderData$treebag_PROB <- predict(object=model_treebag, blenderData[,predictors])
testingData$gbm_PROB <- predict(object=model_gbm, testingData[,predictors])
testingData$rf_PROB <- predict(object=model_rpart, testingData[,predictors])
testingData$treebag_PROB <- predict(object=model_treebag, testingData[,predictors])

# see how each individual model performed on its own
auc <- roc(testingData[,labelName], testingData$gbm_PROB )
print(auc$auc) # Area under the curve: 0.9893

auc <- roc(testingData[,labelName], testingData$rf_PROB )
print(auc$auc) # Area under the curve: 0.958

auc <- roc(testingData[,labelName], testingData$treebag_PROB )
print(auc$auc) # Area under the curve: 0.9734

# run a final model to blend all the probabilities together
predictors <- names(blenderData)[names(blenderData) != labelName]
final_blender_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)

# See final prediction and AUC of blended ensemble
preds <- predict(object=final_blender_model, testingData[,predictors])
auc <- roc(testingData[,labelName], preds)
print(auc$auc)  # Area under the curve: 0.9922
