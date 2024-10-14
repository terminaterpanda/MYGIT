from django.db import models
from konlpy.tag import Okt

# Create your models here.

class Vector(models.Model):
    input_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField("inputed date")

class Input(models.Model):
    inputed_text = models.ForeignKey(Vector, on_delete=models.CASCADE)
    making_text = models.CharField(max_length=200)
    indexing = models.IntegerField(default=0)

from django.db import models

from django.conf import settings
from datetime import date
import numpy as np
import random
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

class TOKENIZER(models.Model):
    name = models.CharField(
        max_length=999,
        unique=True,
        help_text="YOUR TEXT IS INPUTED HERE."
    )
    def __str__(self):
        
        return self.name
    
class LogisticRegressionModel(models.Model):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15  
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict(self, X, w, b):
        z = np.dot(X, w) + b
        return self.sigmoid(z)
    
    #training function
    def train(self, X, y_true, learning_rate = 0.01, iterations = 100):
        w = np.random.randn(X.shape[1])
        b = random.random()

        for i in range(iterations):
            y_pred = self.predict(X, w, b)
            loss = self.cross_entropy_loss(y_true, y_pred)

   
            dw = np.dot(X.T, (y_pred - y_true)) / X.shape[0]
            db = np.sum(y_pred - y_true) / X.shape[0]

        
            w -= learning_rate * dw
            b -= learning_rate * db

        return w, b, loss

    def test(self, X, w, b):
        return self.predict(X, w, b)


# Example of using this model in Django views
def logistic_regression_example():
    # Example data (usually you'd get this from your Tokenizer model)
    X = np.array([[1, 2], [2, 3], [3, 4]])  # Input features
    y_true = np.array([0, 1, 1])  # True labels (binary classification)

    # Initialize and train the logistic regression model
    logistic_model = LogisticRegressionModel()
    w, b, loss = logistic_model.train(X, y_true)

    # Test the model
    predictions = logistic_model.test(X, w, b)

    return {
        'weights': w,
        'bias': b,
        'loss': loss,
        'predictions': predictions
    }
