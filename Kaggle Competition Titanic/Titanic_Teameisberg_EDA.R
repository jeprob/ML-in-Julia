#imports
library(readr)
library(tidyverse)
library(ggfortify)
library(GGally)
library(dplyr)
library(stringr)
library(tidyr)
library(glmulti)

# read in data
train <- read.csv("C:/Users/Jonas/Desktop/Uni/semester_5/Blockkurs_2/titanic_competition/train.csv")
train_onehot <- read.csv("C:/Users/Jonas/train_onehot.csv")
all_onehot <- read.csv("C:/Users/Jonas/all_onehot.csv")
dd <- read.csv("C:/Users/Jonas/Desktop/Uni/semester_5/Blockkurs_2/titanic_competition/train.csv")

#make factors
train$Survived = as.factor(train$Survived)
train$Sex = as.factor(train$Sex)
train$Pclass = as.factor(train$Pclass)
train$Embarked= as.factor(train$Embarked)

# correlations
train_red = select(train, Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

ggpairs(train_red)
#not too many correlated features
onehot_red = select(all_onehot, Survived, Pclass, SibSp, Parch, Fare, Embarked)

#histograms
hist(train$Age, breaks=20) # try out if age classes make more sense and what the range of these classes should be
hist(train$Fare, breaks=100) # same (with potential classes) for the fare

# boxplot of the fare depending on survival
ggplot(train, aes(x=Survived, y=Fare)) + #there is a differnce; survivors payed a higher fare
  scale_y_continuous(limits = c(0, 150))+
  geom_boxplot() 

# boxplot of age depending on survival
ggplot(train, aes(x=Survived, y=Age)) + #not really a difference visible
  scale_y_continuous(limits = c(0, 100))+
  geom_boxplot() 

# barplot of Embarked
ggplot(train, aes(Survived)) +
  geom_bar(aes(fill=Embarked), position = "dodge")
# Q and S both lower #survivors, but C has a higher #survivors than people that died, 1 missing value
# not sure whether to keep Embarked for the algorithm
ggplot(train, aes(Embarked)) +
  geom_bar(aes(fill=Pclass), position = "dodge")

# look at Parch
ggplot(train, aes(Survived)) +
  geom_bar(aes(fill=as.factor(Parch)), position = "dodge")

# effect of deck (first letter in cabin) on survival
ggplot(train_onehot_y, aes(Survived)) +
  geom_bar(aes(fill=Embarked), position = "dodge")


table(train$Sex,train$Survived) #has to be in there
table(train$Survived, train$SibSp)
table(train$Survived, train$Parch) # individuals with Parch > 0 seem to have a higher survival rate
table(train$Survived, train$Embarked)

# conditional density plot of survival depending on age
cdplot(as.factor(Survived)~Age, data=dd)

### bar plots with counts depending on category
# Pclass barplot
ggplot(train, aes(x=as.factor(Survived))) +
  geom_bar(aes(fill = as.factor(Pclass)), position = "dodge")
# clear trend: the better the class, the higher the proportion of survivors

# Sex barplot
ggplot(train, aes(x=as.factor(Survived))) +
  geom_bar(aes(fill = Sex), position = "dodge")

all_onehot$nrTick = as.numeric(all_onehot$nrTick)

all_onehot$Survived = as.factor(all_onehot$Survived)

# ticket number
ggplot(all_onehot,aes(y = nrTick, x = Survived))+
  geom_boxplot()


