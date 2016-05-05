library(MASS)
library(caret)
library(Metrics)

subSpace <- function(x){
  x <- gsub(" ","", x, fixed = T)
  x
}

imputeLikelihood <- function(x,y,all){
  t <- table(x,y)
  t <- t/colSums(t)
  t <- t/rowSums(t)
  
  for(i in levels(x)){
    ind <- which(row.names(t)==i)
    all[all==i] <- t[ind,2]
  }
}

train <- read.csv('train.csv', header = T, stringsAsFactors = F, na.strings = c(NA, ""))
y <- train$target
test <- read.csv('test.csv', header = T, stringsAsFactors = F, na.strings = c(NA, ""))

# Find the column number of target attribute
which(colnames(train)=='target')

# Remove target attribute from train set and combine two sets
all_data <- rbind(train[,-2], test)
charVars <- which(sapply(all_data, is.character))

# Arbitrarily separate categorical variables with more than 15 categories.
# If a categorical variable has less than 16 categories keep it as categorical.
# Otherwise change it into integer.
for (vars in charVars){
  if(length(unique(all_data[,vars]))<16){
    # Encode missing category as -1
    all_data[is.na(all_data[,vars]),vars] <- -1
    all_data[,vars] <- as.factor(all_data[,vars])
  }else{
    all_data[,vars] <- as.integer(as.factor(all_data[,vars]))
  }
}

# Encode NA values as -1
all_data[is.na(all_data)] <- -1

# Find the ones which we kept as categorical
charVars <- which(sapply(all_data, is.factor))

# Make a backup of the data
dat <- all_data
# Generate the formula for dummy encoding. This means that we are going to dummy encode all the remaining categorical vars.
frm <- as.formula(paste("~",paste(colnames(all_data)[charVars], sep="", collapse = "+"), "-1", sep=""))

# Dummy encode using caret package
dummies <- dummyVars(frm, data = all_data, fullRank = T)
dat2 <- predict(dummies, newdata = all_data)

# Combine data and remove redundant categorical vars
dat <- dat[,-charVars]
dat <- data.frame(dat, dat2)
#summary(dat)

# Find near zero variance variables and linear dependent variables.
# Make datasets without those.
nzv <- nearZeroVar(dat)
dat2 <- dat[,-nzv]
comboInfo <- findLinearCombos(dat2)
dat3 <- dat2[,-comboInfo$remove]

train1 <- dat[1:nrow(train),]
test1 <- dat[-(1:nrow(train)),]
train1$target <- y

train2 <- dat2[1:nrow(train),]
test2 <- dat2[-(1:nrow(train)),]
train2$target <- y

train3 <- dat3[1:nrow(train),]
test3 <- dat3[-(1:nrow(train)),]
train3$target <- y

## I initially thought to use t-SNE as well but it turned out to be unnecessary.
#dd <- duplicated(dat)
# 
# library(Rtsne)
# rts <- Rtsne(as.matrix(dat), check_duplicates = FALSE, pca = FALSE, 
#              perplexity=30, theta=0.5, dims=2, verbose = T)
# 
# dat4 <- data.frame(dat, rts$Y)
# train4 <- dat4[1:nrow(train),]
# test4 <- dat4[-(1:nrow(train)),]
# train4$target <- y
# 
# p <- ggplot(aes(x = train4[,4], y = train4[,164], color = as.factor(train4$target)), data = train4)
# p + geom_point()
# 
# write.csv(test4, 'test_numeric_onehot_tsne.csv', row.names = F)
# write.csv(train4, 'train_numeric_onehot_tsne.csv', row.names = F)

# Write a .csv file to be used later in python.
write.csv(train1, "train_numeric_onehot_all.csv", row.names = F)
write.csv(test1, "test_numeric_onehot_all.csv", row.names = F)