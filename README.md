# Heart Disease Prediction Using ANN

## About the Project

I built this project to predict whether a patient is likely to have heart disease using an Artificial Neural Network.

The idea is pretty simple. The model takes information about a patient, such as their age, blood pressure, cholesterol, chest pain type, and maximum heart rate, then uses those features to predict one of two classes:

```text
0 = No Heart Disease
1 = Heart Disease
```

The main goal of this project was to understand how a neural network can be used for a real binary classification problem, while also going through the full process of cleaning the data, preprocessing it, training the model, and evaluating the results.

## Dataset

The dataset originally had 1,025 rows and 13 input features.

Some of the features include:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG results
* Maximum heart rate
* Exercise-induced angina
* ST depression
* ST slope
* Number of major vessels
* Thalassemia result

The target column tells us whether the patient has heart disease.

One thing I noticed while exploring the data was that there were 723 duplicate rows. I removed those before training the model, which left 302 unique patient records.

There were no missing values in the dataset.

## Data Preprocessing

Before training the neural network, I separated the features into numerical and categorical data.

The numerical features were:

```text
age
trestbps
chol
thalach
oldpeak
```

I standardized these features using `StandardScaler` so that features with larger values, such as cholesterol, would not have more influence just because of their scale.

The categorical features were one-hot encoded using `OneHotEncoder`.

These included:

```text
sex
cp
fbs
restecg
exang
slope
ca
thal
```

After one-hot encoding, the original 13 features became 30 inputs for the neural network.

The dataset was then split into 80% training data and 20% testing data.

```text
Training samples: 241
Testing samples: 61
```

## Neural Network

I built the model using TensorFlow and Keras.

The network has two hidden layers:

```text
Input
  |
32 neurons, ReLU
  |
Dropout 0.2
  |
16 neurons, ReLU
  |
Dropout 0.2
  |
1 neuron, Sigmoid
  |
Prediction
```

ReLU was used in the hidden layers, while Sigmoid was used in the output layer.

Sigmoid works well here because this is a binary classification problem. It gives an output between 0 and 1, which can be treated as the probability of heart disease.

For example:

```text
0.20 = 20% probability
0.72 = 72% probability
0.91 = 91% probability
```

The model uses `0.5` as the decision threshold.

```text
Below 0.5 = No Heart Disease
0.5 or above = Heart Disease
```

## Training

The model was trained using:

```text
Optimizer: Adam
Loss Function: Binary Cross-Entropy
Batch Size: 16
Maximum Epochs: 200
```

I also used dropout and early stopping because the dataset is fairly small and the model can easily start overfitting.

Early stopping monitors the validation loss and restores the best version of the model once the validation performance stops improving.

Looking at the training graphs, the model started showing signs of overfitting after around 25 to 30 epochs. Training accuracy continued to improve, while validation performance started to decrease.

## Results

The final model achieved:

```text
Test Accuracy: 80.33%
ROC-AUC: 0.8874
```

The classification results were:

| Class            | Precision | Recall | F1 Score |
| ---------------- | --------: | -----: | -------: |
| No Heart Disease |      0.79 |   0.79 |     0.79 |
| Heart Disease    |      0.82 |   0.82 |     0.82 |

The confusion matrix was:

```text
                 Predicted
                 0     1

Actual 0        22     6
Actual 1         6    27
```

Out of 61 test cases, the model correctly classified 49 and incorrectly classified 12.

I also looked at recall because it is important in this type of problem. A false negative means the model predicts that a patient does not have heart disease when they actually do.

## Predicting New Patients

I also added a function that allows the trained model to make predictions on new patient data.

Example:

```python
predict_heart_disease(
    age=52,
    sex=1,
    cp=2,
    trestbps=130,
    chol=235,
    fbs=0,
    restecg=1,
    thalach=161,
    exang=0,
    oldpeak=0.2,
    slope=2,
    ca=0,
    thal=2
)
```

The model first preprocesses the new patient's data using the same scaler and encoder used during training.

It then returns something like:

```text
Heart Disease Probability: 82.45%
Prediction: Heart Disease
```

## Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Keras
* Matplotlib
* Seaborn
* Jupyter Notebook

## Final Note

This project was mainly built to practice using Artificial Neural Networks for classification.

The model performed reasonably well, but the dataset only contains 302 unique records after removing duplicates. Because of that, this model should only be treated as a learning project and not as a real medical diagnosis system.
