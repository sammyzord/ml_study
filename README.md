# AI/ML study

Result of a ChatGPT assisted study session on AI/ML, the info may be wrong :P but I'll try correcting it when I aquire more knowledge about the subject.

The project in question is about trying to use regression ML algorithms to predict the price of properties in Melbourne using [this dataset I found on Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)

### How to run:
After creating a virualenv, activating, and installing all dependencies. You can run it using the following command:
```
$ python main.py
```

## Classification vs Regression:

- Classification is the task of predicting discrete, categorical labels or classes. It aims to classify data into predefined categories based on its features.
- Regression is the task of predicting continuous, numerical values. It seeks to estimate the relationship between input features and output variables to make predictions.

### Classification Algorithms:

- Logistic Regression: A widely used classification algorithm that models the probability of belonging to different classes based on linear relationships between features. It is interpretable but assumes a linear relationship and is sensitive to outliers.
- Decision Trees: A non-parametric algorithm that splits the data based on features to create a tree-like model. It can handle both numerical and categorical data but tends to overfit the training data.
- Random Forest: A ensemble algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It handles high-dimensional data and provides feature importance but requires parameter tuning.
- Support Vector Machines (SVM): A powerful algorithm that finds the best hyperplane to separate classes in a high-dimensional space. It works well with limited samples but can be computationally expensive and sensitive to kernel choice.

### Regression Algorithms:

- Linear Regression: A simple algorithm that models the linear relationship between input features and continuous output. It is fast and provides interpretable coefficients but assumes linearity and is sensitive to outliers.
- Decision Trees: Similar to classification, decision trees can be used for regression by predicting continuous values instead of classes. They capture nonlinear relationships but tend to overfit the training data.
- Random Forest: Also applicable to regression tasks, random forests reduce overfitting by aggregating multiple decision trees. They handle high-dimensional data and provide feature importance but require parameter tuning.
- Gradient Boosting: An ensemble method that combines weak models (usually decision trees) to build a strong predictive model. It handles different types of data but can be sensitive to outliers and requires careful parameter tuning.

## Imputing:

Handling Missing Data: Various imputation strategies are available in scikit-learn:

- SimpleImputer: Fills missing values with mean, median, most frequent, or a constant value.
- KNNImputer: Estimates missing values using the k-nearest neighbors algorithm.
- IterativeImputer: Uses regression models to impute missing values iteratively.
- MissForest: Imputes missing values using random forests.

## Encoding:

- One-Hot Encoding: Converts categorical variables into binary vectors, creating separate columns for each category. It allows algorithms to work with categorical data, but it increases dimensionality.
- Label Encoding: Assigns unique numerical labels to each category in a categorical variable. It is useful for ordinal data but may introduce a false sense of order in non-ordinal variables.
