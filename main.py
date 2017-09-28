# Imports
import pandas as pd  
import numpy as np

from sklearn.cross_validation import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import precision_recall_curve  
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Retrieve Data Set
df = pd.read_csv('http://www.dataminingconsultant.com/data/churn.txt')


# Some Prepreocessing
df.columns = [c.lower().replace(' ', '_').replace('?', '').replace("'", "") for c in df.columns]

state_encoder = LabelEncoder()  
df.state = state_encoder.fit_transform(df.state)

del df['phone']

binary_columns = ['intl_plan', 'vmail_plan', 'churn']  
for col in binary_columns:  
    df[col] = df[col].map({
            'no': 0
        ,   'False.': 0
        ,   'yes': 1
        ,   'True.': 1
    })


# Build the classifier and get the predictions
clf = RandomForestClassifier(n_estimators=50, oob_score=True)  
test_size_percent = 0.1

signals = df[[c for c in df.columns if c != 'churn']]  
labels = df['churn']

train_signals, test_signals, train_labels, test_labels = train_test_split(signals, labels, test_size=test_size_percent)  
clf.fit(train_signals, train_labels)  
predictions = clf.predict_proba(test_signals)[:,1] 

precision, recall, thresholds = precision_recall_curve(test_labels, predictions)  
thresholds = np.append(thresholds, 1)

queue_rate = []  
for threshold in thresholds:  
    queue_rate.append((predictions >= threshold).mean())

plt.plot(thresholds, precision, color= 'grey')  
plt.plot(thresholds, recall, color= 'red')  
plt.plot(thresholds, queue_rate, color= 'blue')

leg = plt.legend(('precision', 'recall', 'queue_rate'), frameon=True)  
leg.get_frame().set_edgecolor('k')  
plt.xlabel('threshold')  
plt.ylabel('%')
plt.show()