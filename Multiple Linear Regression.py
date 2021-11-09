#MULTIPLE LINEAR REGRESSION USING PANDAS, MATPLOTLIB, SKLEARN, AND NUMPY
#PROGRAM ANALYZES CSV FILE DATA, PREDICTS VALUES, AND GRAPHS THE ERROR
#BUILT ON CODE TAUGHT IN MACHINE LEARNING COURSE BY KIRILL EREMENKO
#MISHA SINITCYN 2021

#IMPORT LIBRARIES
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#IMPORT DATASET
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#ENCODE CATEGORIAL DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#SPLIT INTO TRAINING AND TEST SET
random__state = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random__state)
error_array = [0,0,0,0,0,0,0,0,0,0]

#TRAIN ON TRAIN SET
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

#PRINT PREDICTED VALUES, ACTUAL VALUES, AND MARGIN OF ERROR
for i in range(0,10):
    print( round(y_pred[i],2), round(y_test[i],2), round(abs((y_test[i] - y_pred[i])/y_test[i])*100,2 ), "%" )
    error_array[i-1] += round( abs(((y_test[i] - y_pred[i])/y_test[i]) * 100), 2)

#MARGIN ERROR PER PREDICTED VALUE, AS SHOWN IN GRAPH
print(error_array)
x_axis = [0,1,2,3,4,5,6,7,8,9]

#GRAPH THE ERROR
plt.figure(0)
plt.plot(x_axis, error_array,  color='xkcd:salmon', label='original')
plt.legend()
plt.show()
plt.close(0)
print()
