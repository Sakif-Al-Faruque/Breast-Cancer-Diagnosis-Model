install.packages("corrplot")
install.packages("caret")
install.packages("class")
install.packages("ggplot2")

library(corrplot)
library(caret)
library(class)
library(ggplot2)

project_dataset <- read.csv('/Users/sakif/Desktop/R/ds_project_ft/breast-cancer.csv', 
                            header = TRUE, sep = ',');

str(project_dataset)
summary(project_dataset)
View(project_dataset)
names(project_dataset)
nrow(project_dataset)

project_dataset$diagnosis[
  project_dataset$diagnosis == 'M'
] <- 1
project_dataset$diagnosis[
  project_dataset$diagnosis == 'B'
] <- 0

freqofdia <- table(project_dataset$diagnosis)
View(freqofdia)
png(file = "/Users/sakif/Desktop/R/ds_project_ft/diagnosis_bar_plot_before_cleaning.png")
barplot(freqofdia)
dev.off()

colSums(is.na(project_dataset))
project_dataset <- na.omit(project_dataset)



project_dataset$diagnosis <- as.integer(project_dataset$diagnosis)
cr_ov <- cor(project_dataset)
print(cr_ov)
png(file = "/Users/sakif/Desktop/R/ds_project_ft/total_corelation.png")
corrplot(cr_ov)
dev.off()

project_dataset <- subset(project_dataset, select = -c(id))

project_dataset <- subset(
  project_dataset, 
  select = -c(
    fractal_dimension_mean,
    texture_se,
    smoothness_se,
    symmetry_se,
    fractal_dimension_se
  )
)

cr_ov <- cor(project_dataset)
print(cr_ov)
png(file = "/Users/sakif/Desktop/R/ds_project_ft/total_corelation_first_filter.png")
corrplot(cr_ov)
dev.off()

project_dataset <- subset(
  project_dataset, 
  select = -c(
    perimeter_mean,
    area_mean,
    radius_worst,
    perimeter_worst,
    area_worst,
    texture_worst,
    concavity_worst,
    concave.points_worst
  )
)

cr_ov <- cor(project_dataset)
print(cr_ov)
png(file = "/Users/sakif/Desktop/R/ds_project_ft/total_corelation_after_final_elimination.png")
corrplot(cr_ov)
dev.off()



columns_to_exclude_from_normalization <- c("diagnosis")
columns_to_normalize <- setdiff(names(project_dataset), columns_to_exclude_from_normalization)
project_dataset[columns_to_normalize] <- sapply(project_dataset[columns_to_normalize], function(column) {
  (column - min(column)) / (max(column) - min(column))
})
head(project_dataset)




train_indices <- createDataPartition(project_dataset$diagnosis, p = 0.8, list = FALSE)
train_data <- project_dataset[train_indices, ]
test_data <- project_dataset[-train_indices, ]

train_features <- train_data[, -which(names(train_data) == "diagnosis")]
train_target <- train_data$diagnosis

test_features <- test_data[, -which(names(test_data) == "diagnosis")]
test_target <- test_data$diagnosis



no_of_instances <- nrow(project_dataset)
k <- round(sqrt(no_of_instances), digits = 0)

predicted_labels <- knn(train = train_features, test = test_features, cl = train_target, k = k)
correct_predictions <- sum(predicted_labels == test_target)
total_instances <- length(test_target)
accuracy <- correct_predictions / total_instances
cat("Accuracy with division of data:", accuracy)


train_target <- as.factor(train_target)
num_folds <- 10
train_control <- trainControl(method = "cv", number = num_folds)
knn_model <- train(train_features, train_target, method = "knn", trControl = train_control, 
                   tuneGrid = data.frame(k = k))
cat("Accuracy with 10-fold cross validation:", knn_model$results$Accuracy)



test_target <- as.factor(test_target)
confusion_matrix <- confusionMatrix(predicted_labels, test_target)
print(confusion_matrix)

precision <- confusion_matrix$byClass["Pos Pred Value"]
recall <- confusion_matrix$byClass["Sensitivity"]
cat("Precision:", precision, '\n')
cat("Recall:", recall)
