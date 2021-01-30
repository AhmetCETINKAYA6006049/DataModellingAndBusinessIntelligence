
# create an array of records labels
label <- c(rep(1, 1000), rep(0, 1000))

# tokenization
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, removeNumbers)

# stopwords
corp <- tm_map(corp, removeWords, stopwords("english"))

#Stemming
corp <- tm_map(corp, stemDocument)



# step 3: TF-IDF and latent semantic analysis

# compute TF-IDF
tdm <- TermDocumentMatrix(corp)
tfidf <- weightTfIdf(tdm)

# extract (20) concepts
library(lsa)
lsa.tfidf <- lsa(tfidf, dim = 20)




# convert to data frame
words.df <- as.data.frame(as.matrix(lsa.tfidf$dk))


# sample 60% training data
training <- sample(c(1:2000), 0.6*2000)

# run logistic model on training
trainData = cbind(label = label[training], words.df[training,])
reg <- glm(label ~ ., data = trainData, family = 'binomial')

# compute accuracy on validation set
validData = cbind(label = label[-training], words.df[-training,])
pred <- predict(reg, newdata = validData, type = "response")
View(pred)
# produce confusion matrix
View(label[-training])

View(label[training])
table(ifelse(pred>0.5, 1, 0), label[-training])

