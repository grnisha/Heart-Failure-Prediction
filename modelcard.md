# Model Card: Random Forest Classifier

## Model Details
- **Type**: Random Forest Classifier
- **Framework**: scikit-learn  
- **Version**: 1.0

## Intended Use
- For educational and research purposes only.
- Intended to demonstrate and illustrate machine learning techniques in a controlled, academic setting.
- Not designed for actual clinical diagnosis or medical decision-making.

## Training Data
- Training set: 60% of heart disease dataset.
- Validation set: 20% of heart disease dataset.
- Test set: 20% of heart disease dataset.

## Hyperparameters
- **n_estimators**: 200 
- **max_depth**: 4
- **min_samples_leaf**: 4
- **random_state**: 42

## Evaluation Metrics
- **Confusion Matrix**: Monitored false positives (top right) and false negatives (bottom left).
- **ROC AUC Score**: ROC AUC (Receiver Operating Characteristic Area Under the Curve) measures discrimination capability between positive and negative classes.
- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positive instances.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

## Model Metrics
| Metric              | Value     |
|---------------------|----------:|
| Accuracy (Val)      | 87.50%    |
| Precision (Val)     | 87.62%    |
| Recall (Val)        | 90.20%    |
| F1-Score (Val)      | 88.89%    |
| ROC AUC (Val)       | 0.9200    |

## Performance Overview
- Relatively balanced false positives and false negatives compared to other models.
- May still benefit from further hyperparameter tuning.

## Limitations
- **Data Size**: Only 916 samples were available of which 60% (549) used for training, which limits generalizability.
- **Data Availability**: Some patient subgroups are not fully represented. For example no data available for younger age group (age < 28) and less data for women.
- **Generalization Risk**: Overfitting can occur if hyperparameters are not carefully optimized.
- **Data Bias**: Training data underrepresents certain populations, predictions may not generalize.

## Ethical & Responsible Use
- This is for education purpose only and not intended for clinical use; training data is limited and this model is insufficient for medical diagnosis.
