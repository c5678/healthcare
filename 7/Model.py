# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

USAinsurance = pd.read_csv('Healthcare_cleaned.csv')




X = USAinsurance [['Gender', 'Race', 'Dexa_Freq_During_Rx', 'Dexa_During_Rx', 'Frag_Frac_Prior_Ntm']]
y = USAinsurance ['Persistency_Flag']

X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

pickle.dump(lm, open('model.pickle', 'wb'))