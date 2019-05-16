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
require(ReinforcementLearning)

###Packages for model optimization
require(UBL)  ####for balancing training points
require(caret)##### for evaluating models
require (pROC)
require (caretEnsemble)

####Packages for outlier detection
require(dbscan)
require(fpc)

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
stopCluster(coreNo)

######Load Training Data
#########Always name table as 'X1' for the tests
fplot=(file.choose()) 
X1 <- fread(fplot, stringsAsFactors = FALSE, data.table = FALSE)#

####change the name of the categorical or response variable to be modelled
str (X1) ###examine to find variable
colnames(X1)[1]=c("Class") ## choose column which is the response variable and rename it
colnames(X1)[2]=c("Group") ## choose column which is the grouping variable and rename it
X1$Class <- as.factor(X1$Class)
X1$Class <- gsub("[[:space:]]","",X1$Class) ## for subzone
#########this block of info can be inserted whereever required in the script to split off information/blocks 
str (X1) ###examine to find information of blocking variables
#X1.sub <- X1.sub[-1]
ID_col <- c(2:4) ###columns which are identifiers and not variables
X1_ID <- X1 [, c(ID_col)] ###ID data 
X1.sub <- X1 [, -c(ID_col)] ###Data to be tested

prop.table(table(X1.sub$Class))
#subset the data?
group <- "CWHdmz" ###select a group and reduce dataset to that group
X1.sub <- X1.sub[(X1$Class %in% group),]

# calculate correlation matrix of variables
correlationMatrix <- cor(X1.sub[,-c(1)])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.95, verbose = TRUE, names = TRUE) ## review for removal of highly correlated vars

X1.sub <- X1.sub[, !(colnames(X1.sub) %in% highlyCorrelated), drop = FALSE]
#X1.sub2 <- merge (X1.sub[1], X1.sub1, by = 0, row.names = FALSE)
#X1.sub <- X1.sub2[-1]
descrCor2 <- cor(X1.sub[,-c(1)])
summary(descrCor2[upper.tri(descrCor2)])

##########Assess for training point outliers :dbscan
#Determine the best eps size
dbscan::kNNdistplot(X1.sub[-1], k =  5)
abline(h = 0.15, lty = 2) # chose eps size based on when graphic goes vertical.

set.seed(123)
db <- fpc::dbscan(X1.sub[-1], eps= 100, MinPts =  10)# eps. minimum points
# Plot DBSCAN results
plot(db, X1.sub[-1], main = "DBSCAN", frame = TRUE,  par(pch = 1))
fviz_cluster(db, X1.sub[-1], stand = FALSE,  geom = "point")
dev.off()


########Training Point balancing of uncommon classes caret package
X1.sub2 <- SmoteClassif(Class ~ ., X1.sub, C.perc = "balance", k= 5 , repl = FALSE, dist = "Euclidean") # creates full balanced data set
X1.sub2 <- SmoteClassif(Class ~ ., X1.sub, C.perc = list(E4 = 50, E5 = 100), k= 5 , repl = FALSE, dist = "Euclidean") # creates manusl balancing - C.perc list is variables and weights
prop.table(table(X1.sub2$Class)) ## check percentage of each 
X1.sub <- X1.sub2
    ########Random Over/Under samplin of uncommon classes UBL package - LESS USEFUL
    #X1.sub <- RandUnderClassif(ESuit ~ ., X1.sub)
    #X1.sub <- downSample(x= X1.sub[-1], y= X1.sub$ESuit)

############SPLIT TRAINING SET INTO TRAINING, VALIDATION, TEST SUBSETS

# shuffle and split the data into two parts training and testing (validation data sets can be taken from training in cross validation) 
set.seed(1234)
X1.sub2 <-X1.sub
X1.sub <- X1.sub[-1]
colnames (X1.sub)[1] <- "Class"
X1.sub$Class <- as.factor(X1.sub$Class)
#X1.sub4 <- X1.sub[(X1.sub$Class == "E4"),]
droplevels (X1.sub$Class)
train.list <- createDataPartition(X1.sub$Class, times = 1, p = 0.75, list = FALSE, ##times= number of splits, p= proportion for training set
                                  groups = min(5, length(X1.sub$Class)))

X1.train <- X1.sub [train.list,]
X1.test <- X1.sub [-train.list,]
##########if a separate validation set is desired run the next few lines
train.list2 <- createDataPartition(X1.train$Class, times = 1, p = 0.8, list = FALSE, ##times= number of splits, p= proportion for training set
                                   groups = min(5, length(X1.sub$Class)))
X1.train <- X1.train <- X1.sub [train.list2,]
X1.validate <- X1.train [-train.list2,]

#############Testing for overfitting based on number of variables 
####random forest
set.seed (12345)
control <- trainControl(method= "repeatedcv", number=3, returnResamp = "final",
                        classProbs = TRUE, 
                        search = "random",
                        verboseIter = TRUE,
                        repeats=1,
                        allowParallel = FALSE)#repeats=2, importance  = TRUE, repeats=5,
#Random Forest Model of training data
fit.rf <- train(Class ~.,  X1.train, method='rf', metric="Kappa",   trControl=control, verbose=TRUE,  do.trace = 10)
rf.varimp <- varImp(fit.rf, scale=FALSE)
plot(rf.varimp, top = 21)

 modelcompare <- resamples(list(rf=fit.rf))#c5.0=fit.c50, , tb=fit.tb
modelcompare
summary(modelcompare)
dotplot(modelcompare)

fit.rf <- randomForest(Class ~ ., data=X1.train,  do.trace = 10, mtry= 3,
                       ntree=71, na.action=na.omit, importance=TRUE, proximity=FALSE)
# get predictions for each ensemble model for two last data sets
# and add them back to themselves
X1.train$pred <- predict(fit.rf, X1.train[-1])
missClass <- (sum(X1.train$pred != X1.train$Class))/length(X1.train$Class)

X1.test$pred <- predict(fit.rf, X1.test[-1])
missClass2 <- (sum(X1.test$pred != X1.test$Class))/length(X1.test$Class)
write.csvfit.rf$confusion[, 'class.error']
fit.rf$test



######################################################################
#####test for mtry in rF - can be time consuming

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid", classProbs = TRUE, allowParallel = FALSE)
set.seed(123456)
tunegrid <- expand.grid(.mtry=c(1:10))
X1.sub$Class <- as.factor(X1.sub$Class)
droplevels(X1.sub)
#X1.sub$ESuit <- make.names(X1.sub$ESuit, unique = FALSE, allow_ = TRUE)
rf_gridsearch <- train(Class ~ ., data=X1.sub, method='rf', metric= 'Kappa', tuneGrid=tunegrid,
                       trControl=control,do.trace = 10)
print(rf_gridsearch)
plot(rf_gridsearch)


###########a caret test for optimal training set size using a Learing Curve
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
                              metric = "Kappa", na.omit = TRUE,
                              trControl = trainControl(classProbs = FALSE, method = "cv",
                                                       verbose = TRUE))#, summaryFunction = twoClassSummary))

ggplot(lda_data, aes(x = Training_Size, y = Accuracy, color = Data)) +
  geom_smooth(method = loess, span = .8) +
  theme_bw()

######A CARET test for recursive feature selection RFE
#load up stratification function for script of same name

X2.sub <- X1.sub
 X1.sub <- X1.test
colnames (X1.sub)[1] <-"Class"
#X1.sub <- stratified(X1.sub, group = "Class", size = 10) 
#X1.sub$Class <- gsub("[[:space:]]","",X1.sub$Class) ## for subzone
#group <- "Pl" ###select a group and reduce dataset to that group
#X1.sub <- X1.sub[(X1$Group %in% group),]
X1.sub$Class <- factor(X1.sub$Class, levels=unique(X1.sub$Class))
trainctrl <- trainControl(verboseIter = TRUE)

control <- rfeControl(functions = rfFuncs,
                      #method = "none",
                      #method = "repeatedcv",
                      method = "cv",
                      #repeats = 3,
                      verbose = TRUE,
                      saveDetails = TRUE,allowParallel = FALSE)#
outcomeName <- 'Class'
outcomeName <- as.factor (outcomeName)
X1.sub$Class <- as.factor(X1.sub$Class)
#droplevels(X1.sub)
predictors <- names(X1.sub)[!names(X1.sub) %in% outcomeName]
VariableTest <- rfe(X1.sub[,predictors], X1.sub[,c("Class")],
                    rfeControl = control)
VariableTest
predictors(VariableTest) #top variables
plot (VariableTest, type = c("o", "g")) ###plots the increase in model performance by number of variables

############Feature Selection using Boruta - this 
#https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
require(Boruta)
set.seed(123)
boruta.train <- Boruta(Class~., data = X1.sub, doTrace = 2)
print(boruta.train)
###comparison of variable importance
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
       at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
###Compare tentative variables
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

var_Use <- getSelectedAttributes(final.boruta, withTentative = F)

boruta.df <- attStats(final.boruta)
print(boruta.df)

#######################Compare Models
#https://rdrr.io/cran/caret/man/models.html list available models for Caret
names(getModelInfo()) ###returns names of all possible caret models
methodList = c("rf", "ada", "C50") # create a list of caret models to be run
set.seed (12345)
control <- trainControl(method="repeatedcv", number=5, returnResamp = "final",
                        classProbs = TRUE, 
                        search = "random",
                       
                        allowParallel = FALSE)#repeats=2, importance  = TRUE,

tune.grid <- expand.grid(.cp=0)
# C5.0
X1.sub$Class <- as.factor (X1.sub$Class)
droplevels(X1.sub$Class)
###c50 modle
fit.c50 <- train(Class ~., X1.sub, method="C5.0", metric= "Accuracy", trControl=control,  do.trace = 10)#[,c(1:4)]
#Random Forest Model
fit.rf <- train(Class ~.,  X1.sub, method='rf', metric="Kappa", trControl=control, verbose=TRUE,  do.trace = 10)
rf.varimp <- varImp(fit.rf, scale=FALSE)
plot(rf.varimp, top = 21)
#rpart
fit.tb <- train(Class ~.,  X1.sub, method='treebag', metric="Accuracy", trControl=control, verbose=FALSE)
##staight Random Forest (not caret)
fit.rf2 <- randomForest(Class ~.,  X1.sub, importance= TRUE)
# compare short list of models
modelcompare <- resamples(list(c5.0=fit.c50, rf=fit.rf, tb=fit.tb))
modelcompare
summary(modelcompare)
dotplot(modelcompare)

#############a look at variable importance

fit.rf_imp <- as.data.frame( fit.rf2$importance )
fit.rf_imp$features <- rownames(fit.rf_imp)
fit.rf_imp_sorted <- arrange( fit.rf_imp , desc(MeanDecreaseAccuracy)  )
barplot(fit.rf_imp_sorted$MeanDecreaseAccuracy, ylab="Variable Importance")
###return only top features
topvar <- X1.sub[ , fit.rf_imp_sorted[1:16,"features"] ]  
X1.sub3 <- merge (X1.sub[1], topvar, by =0)
X1.sub3 <- X1.sub3[-1]
fit.rf_new <- randomForest(Class ~ . , X1.sub3 , ntree=71, importance = TRUE )  



#############Long list taken from https://machinelearningmastery.com/evaluate-machine-learning-algorithms-with-r/
    #Linear Methods
# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(diabetes~., data=dataset, method="lda", metric=metric, preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
fit.glm <- train(diabetes~., data=dataset, method="glm", metric=metric, trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(diabetes~., data=dataset, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
    ####Non-linear methods
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(diabetes~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(diabetes~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Naive Bayes
set.seed(seed)
fit.nb <- train(diabetes~., data=dataset, method="nb", metric=metric, trControl=control)
    ##########Trees and Rules
# CART
set.seed(seed)
fit.cart <- train(diabetes~., data=dataset, method="rpart", metric=metric, trControl=control)

    #######Ensembles of Trees
# C5.0
set.seed(seed)
fit.c50 <- train(diabetes~., data=dataset, method="C5.0", metric=metric, trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(diabetes~., data=dataset, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(diabetes~., data=dataset, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(diabetes~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)

results <- resamples(list(lda=fit.lda, logistic=fit.glm, glmnet=fit.glmnet,
                          svm=fit.svmRadial, knn=fit.knn, nb=fit.nb, cart=fit.cart, c50=fit.c50,
                          bagging=fit.treebag, rf=fit.rf, gbm=fit.gbm))
# Table comparison of long list
summary(results)
# boxplot comparison
bwplot(results)
# Dot-plot comparison
dotplot(results)

##############################################

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
methodList = c("rf", "ada", "C50") # create a list of caret models to be run

model_list <- caretList(
  Class~., data=X1.sub,
  trControl=my_control,
  methodList= methodList
)


# create a caret control object to control the number of cross-validations performed
myControl <- trainControl(method='cv', number=3, returnResamp='none')

# quick benchmark model 
train_model <- train(X1.train[,predictors], X1.train$Class, method='rf', trControl=myControl)
test_model <- train(X1.validate[,predictors], X1.validate$Class, method='rf', trControl=myControl)
preds <- predict(object=train_model, X1.test[,predictors], type = "raw")
preds <- ordered(preds, levels = c("E1", "E2", "E3", "E4", "E5"))

library(pROC)
auc <- multiclass.roc(X1.test[,labelName], preds)
print(auc$auc) # Area under the curve: 0.9896

# train all the ensemble models with ensembleData
model_gbm <- train(X1.train[,predictors], X1.train[,labelName], method='gbm', trControl=myControl)
model_rpart <- train(X1.train[,predictors], X1.train[,labelName], method='rpart', trControl=myControl)
model_treebag <- train(X1.train[,predictors], X1.train[,labelName], method='treebag', trControl=myControl)

# get predictions for each ensemble model for two last data sets
# and add them back to themselves
X1.validate$gbm_PROB <- predict(object=model_gbm, X1.validate[,predictors])
X1.validate$rf_PROB <- predict(object=model_rpart, X1.validate[,predictors])
X1.validate$treebag_PROB <- predict(object=model_treebag, X1.validate[,predictors])
X1.test$gbm_PROB <- predict(object=model_gbm, X1.test[,predictors])
X1.test$gbm_PROB <- ordered(preds, levels = c("E1", "E2", "E3", "E4", "E5"))
X1.test$rf_PROB <- predict(object=model_rpart, X1.test[,predictors])
X1.test$rf_PROB <- ordered(preds, levels = c("E1", "E2", "E3", "E4", "E5"))
X1.test$treebag_PROB <- predict(object=model_treebag, X1.test[,predictors])
X1.test$treebag_PROB <- ordered(preds, levels = c("E1", "E2", "E3", "E4", "E5"))
# see how each individual model performed on its own
auc <- multiclass.roc(X1.test[,labelName], X1.test$gbm_PROB )
print(auc$auc) # Area under the curve: 0.9893

auc <- multiclass.roc(X1.test[,labelName], X1.test$rf_PROB )
print(auc$auc) # Area under the curve: 0.958

auc <- multiclass.roc(X1.test[,labelName], X1.test$treebag_PROB )
print(auc$auc) # Area under the curve: 0.9734

# run a final model to blend all the probabilities together
predictors <- names(X1.validate)[names(X1.validate) != labelName]
final_blender_model <- train(X1.validate[,predictors], X1.validate[,labelName], method='gbm', trControl=myControl)

# See final prediction and AUC of blended ensemble
preds <- predict(object=final_blender_model, X1.test[,predictors])
preds <- ordered(preds, levels = c("E1", "E2", "E3", "E4", "E5"))
auc <- multiclass.roc(X1.test[,labelName], preds)
print(auc$auc)  # Area under the curve: 0.9922


stopImplicitCluster()



############


# Example of Stacking algorithms
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)
# correlation between results
modelCor(results)
splom(results)

### from https://machinelearningmastery.com/machine-learning-ensembles-with-r/
## combine models using caretEnsemble::caretStack

# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# stack using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)


##########Tests of Precision and Recall

