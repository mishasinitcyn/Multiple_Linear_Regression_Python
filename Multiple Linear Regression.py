#IMPORTING LIBRARIES
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#IMPORTING DATASET
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#ENCODE CATEGORIAL DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


for j in range(0,10):
    #SPLIT INTO TRAINING AND TEST SET
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = j)

    #TRAIN ON TRAIN SET
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    #PREDICTING THE TEST SET RESULTS
    y_pred = regressor.predict(X_test)

    a_error = 0
    error_array = [0,0,0,0,0,0,0,0,0,0]

    for i in range(0,10):
        print( round(y_pred[i],2), round(y_test[i],2), round((y_pred[i]/y_test[i])*100,2 ) )
        a_error += (y_pred[i]/y_test[i])**2
        #error_array[i-1] += round((y_pred[i]/y_test[i])**2 * 100, 2)
        error_array[i-1] += round( abs((y_pred[i]/y_test[i]) * 100 - 100), 2)
        
    print(error_array)
    x_axis = [0,1,2,3,4,5,6,7,8,9]

    plt.figure(j)
    plt.plot(x_axis, error_array,  color='xkcd:salmon', label='original')
    plt.legend()
    plt.show()
    plt.close(j)

    print()
    print(a_error)
