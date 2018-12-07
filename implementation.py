# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Nursery.csv', header = None)
transactions = []


for i in range(0, 9411):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 9)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.04, min_confidence = .95, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

print(results)



