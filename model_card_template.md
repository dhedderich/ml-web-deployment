# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
As a model an Random forrest classifier is used as it achieves quite good classification metrics across the board of multiple use cases. The Random forrest is using the standard initialization in terms of hyperparameters and there is no tuning applied.

## Intended Use
The model should be used to classify if the salary level of a person is higher or lower than $50.000 where > equals 1 as a label.

## Training Data
The training data consists of the following columns:
age,
workclass,
fnlgt,
education,
education-num,
marital-status,
occupation,
relationship,
race,
sex,
capital-gain,
capital-loss,
hours-per-week,
native-country,
salary

Before the data is used for training it is split into training and test set on a 80/20 ratio.

The label "salary" is transformed with a binary label encoder and all categorical columns are encoded with One Hot Encoding.

## Evaluation Data
The evaluation data is the before mentioned test set. It is transformed using the created encoders (Label Encoder, One Hot Encoder) to transform it into the correct shape.

## Metrics
The following metrics were used with the appropriate result:
Precision: 0.73
Recall: 0.63
fbeta: 0.68

## Ethical Considerations
Please only use this model on customer data that gave an appropriate consent for such classification in terms of GDPR. The model is prone to errors and as it is obvious from the metrics the model can make wrong predictions, that should not have an effect on the respective customer.

## Caveats and Recommendations
For further use the model training should be optimized with further hyperparameter tuning and additional models could be tried out like XGBoost or other more advanced ensemble methods.
