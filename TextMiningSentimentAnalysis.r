# From: smith@logos.asd.sgi.com (Tom Smith) Subject: Ford Explorer 4WD -
# do I need performance axle?
# We’re considering getting a Ford Explorer XLT with 4WD and we have the
# following questions (All we would do is go skiing - no off-roading):
# 1. With 4WD, do we need the “performance axle” - (limited slip axle). Its
# purpose is to allow the tires to act independently when the tires are on
# different terrain.
# 2. Do we need the all-terrain tires (P235/75X15) or will the all-season
# (P225/70X15) be good enough for us at Lake Tahoe?
# Thanks,
# Tom
# –
# ================================================
# Tom Smith Silicon Graphics smith@asd.sgi.com 2011 N. Shoreline Rd. MS
# 8U-815 415-962-0494 (fax) Mountain View, CA 94043
# ================================================

#install.packages("tm")
library(tm)
# step 1: import and label records
# read zip file into a corpus
corp <- Corpus(ZipSource("E:/AutoAndElectronics.zip", recursive = T))
# create an array of records labels
label <- c(rep(1, 1000), rep(0, 1000))
label
# step 2: text preprocessing
# tokenization
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, removeNumbers)


# stopwords
#install.packages("SnowballC")
corp <- tm_map(corp, removeWords, stopwords("english"))
# stemming
corp <- tm_map(corp, stemDocument)

# compute TF-IDF
tdm <- TermDocumentMatrix(corp)
tdm
tfidf <- weightTfIdf(tdm)
tfidf

# extract (20) concepts
install.packages("lsa")
library(lsa)
lsa.tfidf <- lsa(tfidf, dim = 20)
# convert to data frame
words.df <- as.data.frame(as.matrix(lsa.tfidf$dk))
words.df

set.seed(123)
# sample 60% training data
training <- sample(c(1:2000), 0.6*2000)

# run logistic model on training
trainData = cbind(label = label[training], words.df[training,])
head(trainData)
reg <- glm(label ~ ., data = trainData, family = 'binomial')
summary(reg)

# compute accuracy on validation set
validData = cbind(label = label[-training], words.df[-training,])
pred <- predict(reg, newdata = validData, type = "response")
summary(pred)
str(pred)
View(pred)
str(label[-training])
View(label[-training])
# # produce confusion matrix
# library(caret)
# table(ifelse(pred>0.5, 1, 0), label[-training])
# 774/nrow(validData)

library(caret)
confusionMatrix(ifelse(pred>0.5, 1, 0), label[-training])


