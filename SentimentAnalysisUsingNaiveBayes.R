install.packages("RTextTools")
library(RTextTools)
library(e1071)
library(caret) 

sentPositive = c(
  "I like it", "like it a lot", "It's really good",
  "recommend!", "Enjoyed!", "like it",
  "It's really good", "recommend too",
  "outstanding", "good", "recommend!",
  "like it a lot", "really good", 
  "Definitely recommend!", "It is fun",
  "liked!", "highly recommend this",
  "fantastic show", "exciting",
  "Very good", "it's ok",
  "exciting show", "amazing performance",
  "it is great!","I am excited a lot",
  "it is terrific", "Definitely good one",
  "very satisfied", "Glad we went",
  "Once again outstanding!", "awesome"
)

sentNegative = c(
  "Not good at all!", "rude",
  "It is rude", "I don't like this type",
  "poor", "Boring", "Not good!",
  "not liked", "I hate this type of",
  "not recommend", "not satisfied",
  "not enjoyed", "Not recommend this.",
  "disgusting movie","waste of time",
  "feel tired after watching this",
  "horrible performance", "not so good",
  "so boring I fell asleep", "poor show",
  "a bit strange","terrible"
)

df = data.frame(sentiment = "positive", text = sentPositive)
df = rbind(df, data.frame(sentiment = "negative", text = sentNegative))

# split the df data into the train and test parts.
index = sample(1:nrow(df), size = .9 * nrow(df))
train = df[index, ]
test = df[-index, ]


# create matrix data from the text of a train and test data 
# with a create_matrix function of the RTextTool package. 
# The RTextTool is a package for text classification
mTrain = create_matrix(train[,2], language="english", 
                       removeStopwords=FALSE, removeNumbers=TRUE, 
                       stemWords=FALSE) 
matTrain = as.matrix(mTrain)

mTest = create_matrix(test[,2], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=FALSE) 
matTest = as.matrix(mTest)

print(matTest)


# create the classifier model with NaiveBayes algorithm
labelTrain = as.factor(train[,1])
labelTest = as.factor(test[,1])

model = naiveBayes(matTrain, labelTrain)

# evaluate the fitted model.
pred = predict(model, matTrain) 
confusionMatrix(labelTrain, pred)

# predict our test data with the fitted model and check the accuracy.
pred = predict(model, matTest); 
data.frame(test,pred)
confusionMatrix(labelTest, pred)






