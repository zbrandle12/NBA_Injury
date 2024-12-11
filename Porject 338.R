library(ggcorrplot)
library(readxl)
library(ggplot2)
library(dplyr)
library(corrplot)
library(e1071)
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot)
library(glmnet)
library(tree)


nba_data = read_excel("/Users/zacharybrandle/Downloads/NBA_Injury.xlsx")
head(data)

sum(nba_data$`DAYS MISSED` == 0)
nba_data_injured <- nba_data %>% filter(`DAYS MISSED` > 0)

# Summary of Key Variables
summary(nba_data)
shapiro_test <- shapiro.test(nba_data_injured$`DAYS MISSED`)
print(shapiro_test)
qqnorm(nba_data_injured$`DAYS MISSED`)
qqline(nba_data_injured$`DAYS MISSED`, col = "red")
ggplot(nba_data_injured, aes(x = `DAYS MISSED`)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", alpha = 0.6) +
  geom_density(color = "red", size = 1) +
  labs(title = "Histogram of Days Missed with Density Curve", x = "Days Missed", y = "Density") +
  theme_minimal()
# Distribution of Days Missed
ggplot(nba_data_injured, aes(x = `DAYS MISSED`)) +
  geom_histogram(bins = 20, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Days Missed Due to Injury")

# Boxplot by Injury Type
ggplot(nba_data_injured, aes(x = INJURED_TYPE, y = `DAYS MISSED`)) +
  geom_boxplot(fill = "orange", alpha = 0.6) +
  labs(title = "Days Missed by Injury Type", x = "Injury Type", y = "Days Missed") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#bar graph
nba_data_injured %>%
  group_by(INJURED_TYPE) %>%
  summarise(mean_days_missed = mean(`DAYS MISSED`, na.rm = TRUE)) %>%
  ggplot(aes(x = INJURED_TYPE, y = mean_days_missed)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(title = "Average Days Missed by Injury Type (Bar Graph)", x = "Injury Type", y = "Average Days Missed") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Age vs. Days Missed
ggplot(nba_data_injured, aes(x = AGE, y = `DAYS MISSED`)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Age vs. Days Missed")

# Height vs. Days Missed
ggplot(nba_data_injured, aes(x = PLAYER_HEIGHT_INCHES, y = `DAYS MISSED`)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "purple") +
  labs(title = "Player Height vs. Days Missed")

# Minutes Played (MIN) vs. Days Missed
ggplot(nba_data_injured, aes(x = MIN, y = `DAYS MISSED`)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Minutes Played vs. Days Missed")



avg_days_missed_by_position <- nba_data_injured %>%
  group_by(Position) %>%
  summarise(avg_days_missed = mean(`DAYS MISSED`, na.rm = TRUE),
            count = n()) 
ggplot(avg_days_missed_by_position, aes(x = Position, y = avg_days_missed, fill = Position)) +
  geom_bar(stat = "identity") +
  labs(title = "Average Days Missed by Position",
       x = "Position",
       y = "Average Days Missed") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



mean(nba_data_injured$AGE)
mean(nba_data_injured$PLAYER_HEIGHT_INCHES)
mean(nba_data_injured$PLAYER_WEIGHT)
mean(nba_data_injured$`DAYS MISSED`)

numeric_data <- nba_data_injured %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_data, use = "complete.obs")

correlation_df = data.frame(
  Variable = names(sorted_correlation),
  Correlation = sorted_correlation
)
ggplot(correlation_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Correlation of DAYS MISSED with Other Variables", x = "Variables", y = "Correlation") +
  theme_minimal()

########## MODEL BUILDING LINEAR FIRST
set.seed(123)
train_index = sample(1:nrow(nba_data_injured), 0.7 * nrow(nba_data_injured))
train_data = nba_data_injured[train_index, ]
test_data = nba_data_injured[-train_index, ]
model =  lm(`DAYS MISSED` ~ AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + MIN + GP + PACE + USG_PCT + DRIVES + AVG_SPEED + TOUCHES, data = train_data)
summary(model)
model2 =  lm(`DAYS MISSED` ~   MIN + GP + PACE  + MIN * PACE, data = train_data)
summary(model2)

fullmodel =  lm(`DAYS MISSED` ~ AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + MIN + GP + PACE + USG_PCT + DRIVES + AVG_SPEED + TOUCHES, data = train_data)
null_model = lm(`DAYS MISSED` ~ 1, data = train_data)  
stepwise_model = step(null_model, 
                       scope = list(lower = null_model, upper = fullmodel),
                       direction = "both",
                       trace = TRUE)
test_data$predicted1 = predict(stepwise_model, newdata = test_data)
rmse1 = sqrt(mean((test_data$predicted1 - test_data$`DAYS MISSED`)^2))
cat("RMSE:", rmse1, "\n")
mae1 = mean(abs(test_data$predicted1 - test_data$`DAYS MISSED`))
cat("MAE:", mae1, "\n")

test_data$predicted = predict(model2, newdata = test_data)
rmse = sqrt(mean((test_data$predicted - test_data$`DAYS MISSED`)^2))
cat("RMSE:", rmse, "\n")
mae = mean(abs(test_data$predicted - test_data$`DAYS MISSED`))
cat("MAE:", mae, "\n")


###### RIDGE AND LASSO
set.seed(123)
x <- model.matrix(`DAYS MISSED` ~ MIN + GP + PACE + MIN * PACE, data = train_data)[, -1]
y <- train_data$`DAYS MISSED`
x_test <- model.matrix(`DAYS MISSED` ~ MIN + GP + PACE + MIN * PACE, data = test_data)[, -1]
y_test <- test_data$`DAYS MISSED`
ridge_model <- cv.glmnet(x, y, alpha = 0) 
plot(ridge_model)
(ridge_lambda <- ridge_model$lambda.min)
ridge_pred <- predict(ridge_model, newx = x_test, s = ridge_lambda)
ridge_rmse <- sqrt(mean((ridge_pred - y_test)^2))
ridge_mae <- mean(abs(ridge_pred - y_test))
cat("Ridge RMSE:", ridge_rmse, "\n")
cat("Ridge MAE:", ridge_mae, "\n")

set.seed(123)
lasso_model <- cv.glmnet(x, y, alpha = 1)  
plot(lasso_model)
(lasso_lambda <- lasso_model$lambda.min)
lasso_coef <- coef(lasso_model, s = "lambda.min")
print(lasso_coef)
lasso_pred <- predict(lasso_model, newx = x_test, s = "lambda.min")
lasso_rmse <- sqrt(mean((lasso_pred - y_test)^2))
lasso_mae <- mean(abs(lasso_pred - y_test))
cat("LASSO RMSE:", lasso_rmse, "\n")
cat("LASSO MAE:", lasso_mae, "\n") 
# RMSE of 18.68 but MAE of 7.51 which is almost identical to our linear mode; 




nba_data_injured <- nba_data %>% filter(`DAYS MISSED` > 0)
numeric_data <- nba_data_injured %>% select_if(is.numeric)

set.seed(786)
train_indices <- sample(1:nrow(numeric_data), 0.8 * nrow(numeric_data))
train_data <- numeric_data[train_indices, ]
test_data <- numeric_data[-train_indices, ]

x_train <- train_data %>% select(-`DAYS MISSED`)
y_train <- train_data$`DAYS MISSED`
x_test <- test_data %>% select(-`DAYS MISSED`)
y_test <- test_data$`DAYS MISSED`

svr_model <- svm(x = x_train, y = y_train, type = "eps-regression", kernel = "radial")

summary(svr_model)

y_pred <- predict(svr_model, x_test)

mae <- mean(abs(y_test - y_pred))
mae

rmse <- sqrt(mean((y_test - y_pred)^2))
rmse

library(ggplot2)
results <- data.frame(Actual = y_test, Predicted = y_pred)
ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "solid") +
  labs(title = "SVM Actual vs Predicted Values (DAYS MISSED)",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal()

tuned_model <- tune(
  svm,
  train.x = x_train,
  train.y = y_train,
  kernel = "radial",
  ranges = list(cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1))
)

best_model <- tuned_model$best.model
summary(best_model)







set.seed(18)
nbaidx=sample(1:nrow(nba_data_injured), nrow(nba_data_injured)*0.7)
NBAtrn <- nba_data_injured[nbaidx,]
NBAtst <- nba_data_injured[-nbaidx,]

NBA_tree1 <- tree(`DAYS MISSED` ~ Position + AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + USG_PCT + PACE + DRIVES + TOUCHES, data=NBAtrn)
summary(NBA_tree1)
plot(NBA_tree1)
text(NBA_tree1,pretty=0)
NBApredtrn = predict(NBA_tree1, newdata = NBAtrn)
(maetrn <- mean(abs(NBApredtrn-NBAtrn$`DAYS MISSED`)))
(rmsetrn = sqrt(mean((NBApredtrn-NBAtrn$`DAYS MISSED`)^2)))
NBApredtst = predict(NBA_tree1, newdata = NBAtst)
(maetst <- mean(abs(NBApredtst-NBAtst$`DAYS MISSED`)))
(rmsetst = sqrt(mean((NBApredtst-NBAtst$`DAYS MISSED`)^2)))

# first tree has MAE of 9.96 and RMSE of 23.54


NBA_tree2=rpart(`DAYS MISSED` ~ Position + AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + USG_PCT + PACE + DRIVES + TOUCHES, data=NBAtrn, method="anova")
summary(NBA_tree2)
rpart.plot(NBA_tree2)


NBAtreeprune = prune.tree(NBA_tree1, best = 3)
summary(NBAtreeprune)
plot(NBAtreeprune)
text(NBAtreeprune, pretty = 0)
title(main = "Pruned Regression Tree")
NBA_trn_prune = predict(NBAtreeprune, newdata = NBAtrn)
(maetrn <- mean(abs(NBA_trn_prune-NBAtrn$`DAYS MISSED`)))
(rmsetrn = sqrt(mean((NBA_trn_prune-NBAtrn$`DAYS MISSED`)^2)))
NBA_tst_prune = predict(NBAtreeprune, newdata = NBAtst)
(maetst <- mean(abs(NBA_tst_prune-NBAtst$`DAYS MISSED`)))
(rmsetst = sqrt(mean((NBA_tst_prune-NBAtst$`DAYS MISSED`)^2)))


(bag.NBA = randomForest(`DAYS MISSED` ~ Position + AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + USG_PCT + PACE + DRIVES + TOUCHES,data = NBAtrn, mtry = 13, importance = TRUE))
yhat.bagtrn = predict(bag.NBA,newdata = NBAtrn)
yhat.bagtst = predict(bag.NBA,newdata = NBAtst)
plot(yhat.bagtst,NBAtst$`DAYS MISSED`)
abline(0,1)
(maetrnbag <- mean(abs(yhat.bagtrn - NBAtrn$`DAYS MISSED`)))
(rmsetrnbag = sqrt(mean((yhat.bagtrn - NBAtrn$`DAYS MISSED`)^2)))
(maetstbag <- mean(abs(yhat.bagtst - NBAtst$`DAYS MISSED`)))
(rmsetstbag = sqrt(mean((yhat.bagtst - NBAtst$`DAYS MISSED`)^2)))
importance(bag.NBA)
varImpPlot(bag.NBA)

(bag.NBA2 = randomForest(`DAYS MISSED` ~ Position + AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + USG_PCT + PACE + DRIVES + TOUCHES,data = NBAtrn, mtry = 13, importance = TRUE,ntree = 100))
yhat.bag2 = predict(bag.NBA2,newdata = NBAtst)
plot(yhat.bag2,NBAtst$`DAYS MISSED`)
abline(0,1)
(maetst <- mean(abs(yhat.bag2 - NBAtst$`DAYS MISSED`)))
(rmsetst = sqrt(mean((yhat.bag2 - NBAtst$`DAYS MISSED`)^2)))
importance(bag.NBA2)
varImpPlot(bag.NBA2)

(rf.NBA = randomForest(`DAYS MISSED` ~ Position + AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + USG_PCT + PACE + DRIVES + TOUCHES,data = NBAtrn, mtry =3, importance = TRUE))
plot(rf.NBA)
yhat.rftrn = predict(rf.NBA,newdata = NBAtrn)
yhat.rftst = predict(rf.NBA,newdata = NBAtst)
plot(yhat.rftst,NBAtst$`DAYS MISSED`)
abline(0,1)
(maetrn <- mean(abs(yhat.rftrn - NBAtrn$`DAYS MISSED`)))
(rmsetrn = sqrt(mean((yhat.rftrn - NBAtrn$`DAYS MISSED`)^2)))
(maetst <- mean(abs(yhat.rftst - NBAtst$`DAYS MISSED`)))
(rmsetst = sqrt(mean((yhat.rftst - NBAtst$`DAYS MISSED`)^2)))
importance(rf.NBA)
varImpPlot(rf.NBA)


set.seed(18)
NBA_idx = sample(1:nrow(nba_data_injured), nrow(nba_data_injured) / 2)
NBAtrn2 = nba_data_injured[NBA_idx,]
NBAtst2 = nba_data_injured[-NBA_idx,]

(boost.NBA = gbm(`DAYS MISSED` ~ AGE + PLAYER_HEIGHT_INCHES + PLAYER_WEIGHT + USG_PCT + PACE + DRIVES + TOUCHES,data = NBAtrn2, distribution  ="gaussian", n.trees = 5000,interaction.depth = 4,shrinkage=0.05))
summary(boost.NBA)
yhat.boost = predict(boost.NBA, newdata = NBAtst2, n.trees=5000)
(maetst <- mean(abs(yhat.boost - NBAtst2$`DAYS MISSED`)))
(rmsetst = sqrt(mean((yhat.boost - NBAtst2$`DAYS MISSED`)^2)))





