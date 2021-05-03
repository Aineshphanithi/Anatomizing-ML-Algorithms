# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 01:13:51 2021

@author: B LEKH RAJ
"""
class dataPreprocessing:
    
    def __init__(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def missingData(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
        
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(X[:, 1:3])
        X[:, 1:3] = imputer.transform(X[:, 1:3])
    
    def encodingIndependentVariables(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
        
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
    
    def encodingDependentVariables(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    
    def featureScaling(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
        X_test[:, 3:] = sc.transform(X_test[:, 3:])

class regression:
    
    def __init__(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    # def simpleLinearRegression(self,dir):
    #
    #     # Importing the dataset
    #     dataset = pd.read_csv(dir)
    #     X = dataset.iloc[:, :-1].values
    #     y = dataset.iloc[:, -1].values
    #
    #     # Splitting the dataset into the Training set and Test set
    #     from sklearn.model_selection import train_test_split
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    #
    #     # Training the Simple Linear Regression model on the Training set
    #     from sklearn.linear_model import LinearRegression
    #     regressor = LinearRegression()
    #     regressor.fit(X_train, y_train)
    #
    #     # Predicting the Test set results
    #     y_pred = regressor.predict(X_test)
    

    def multipleLinearRegression(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Training the Multiple Linear Regression model on the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # from sklearn.metrics import confusion_matrix, accuracy_score
        
        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        #accScore = accuracy_score(y_test, y_pred)
        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        # plt.show()
        # print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("multiple Linear Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]

    def polynomialRegression(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
         # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        print(y)
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        

        # Training the Polynomial Regression model on the whole dataset
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(X_train)
        from sklearn.linear_model import LinearRegression
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y_train)
        
        # Predicting a new result with Polynomial Regression
        y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))


        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        # plt.show()
        # print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]

        
    def supportVectorRegression(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        print(X)
        print(y)
        y = y.reshape(len(y),1)
        print(y)
         # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        print(y)
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        
        # Splitting the dataset into the Training set and Test set
        

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)
        print(X)
        print(y)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Training the SVR model on the whole dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        print("support vector Regression")
        # Predicting a new result
        #sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
        # Predicting a new result with Polynomial Regression
        y_pred = sc_y.inverse_transform(regressor.predict(X_test))
        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        plt.show()
        print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]


    def decisionTreeRegression(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
         # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        print(y)
        
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Training the Decision Tree Regression model on the whole dataset
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(X_train, y_train)
        print("descision tree Regression")
        # Predicting a new result
        #regressor.predict([[6.5]])

        # Predicting a new result with Polynomial Regression
        y_pred = regressor.predict(X_test)


        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        # plt.show()
        # print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]

        
    def randomForestRegression(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
         # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        print(y)
        
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        

        # Training the Random Forest Regression model on the whole dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X_train, y_train)
        print("random forest Regression")
        # Predicting a new result
        #regressor.predict([[6.5]])

        # Predicting a new result with Polynomial Regression
        y_pred = regressor.predict(X_test)


        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        # plt.show()
        # print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]



    def xgBoostR(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        #https://www.geeksforgeeks.org/xgboost-for-regression/
        # Necessary imports
        import xgboost as xg
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error as MSE
        
        print("Load the data")
        dataset = pd.read_csv(dir)
        X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
          
        print("Encoding categorical data")
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        
        print("removing the string from independent variable")
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        print(y)
        
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            print("removing the string from dependent variable")
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)


        print("Splitting")
        train_X, test_X, train_y, y_test = train_test_split(X, y,test_size = 0.3, random_state = 123)
          
        print("Instantiation")
        xgb_r = xg.XGBRegressor(objective ='reg:linear',
                          n_estimators = 10, seed = 123)
          
        # Fitting the model
        xgb_r.fit(train_X, train_y)
          
        # Predict the model
        y_pred = xgb_r.predict(test_X)
        print("xgboost Regression")
        # RMSE Computation
        rmse = np.sqrt(MSE(y_test, y_pred))
        print("RMSE : % f" %(rmse))

        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        # plt.show()
        # print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]



    def catBoostR(self,dir):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        #https://catboost.ai/docs/concepts/python-usages-examples.html
        from catboost import CatBoostRegressor
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        from sklearn.model_selection import train_test_split
        train_X, test_X, train_y, y_test = train_test_split(X, y,test_size = 0.3, random_state = 123)
        
        # Initialize CatBoostRegressor
        model = CatBoostRegressor(iterations=2,
                                  learning_rate=1,
                                  depth=2)
        # Fit model
        model.fit(train_X, train_y)
        print("cat boost Regression")
        # Get predictions
        y_pred = model.predict(test_X)

        np.set_printoptions(precision=2)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        # plt.show()
        # print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse]


    def graphs(self):
        #Under Construction
        print("hello")

class classification:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def logisticRegression(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
  
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the Logistic Regression model on the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
    def kNearestNeighbors(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
   
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the K-NN model on the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

    def supportVectorMachine(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
     
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the SVM model on the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

    def kernelSupportVectorMachine(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the Kernel SVM model on the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

    def naiveBayes(self,dir):
 
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the Naive Bayes model on the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
    
    def decisionTreeClassification(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
       
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the Decision Tree Classification model on the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

    def randomForestClassification(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
       
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        print(X_train)
        print(X_test)
        
        # Training the Random Forest Classification model on the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
    def xgBoostC(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Training XGBoost on the Training set
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
        # Applying k-Fold Cross Validation
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
        
    def catBoostC(self,dir):
        
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        print(y)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        from catboost import CatBoostClassifier
        
        classifier = CatBoostClassifier()
        classifier.fit(X_train, y_train)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
        # Applying k-Fold Cross Validation
        from sklearn.model_selection import cross_val_score
        
        accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
        print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
        print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
        
class clustering:
    
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def k_MeansClustering(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        # Using the elbow method to find the optimal number of clusters
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        
        # Training the K-Means model on the dataset
        kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)

    def hierarchicalClustering(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        
        # Using the dendrogram to find the optimal number of clusters
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()
        
        # Training the Hierarchical Clustering model on the dataset
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        
class associationRuleLearning:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def apriori(self,dir):
        
        # Data Preprocessing
        dataset = pd.read_csv(dir, header = None)
        transactions = []
        for i in range(0, 7501):
          transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
        
        # Training the Apriori model on the dataset
        from apyori import apriori
        rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
        
        # Visualising the results
        
        ## Displaying the first results coming directly from the output of the apriori function
        results = list(rules)
        print(results)
        
        ## Putting the results well organised into a Pandas DataFrame
        def inspect(results):
            lhs         = [tuple(result[2][0][0])[0] for result in results]
            rhs         = [tuple(result[2][0][1])[0] for result in results]
            supports    = [result[1] for result in results]
            confidences = [result[2][0][2] for result in results]
            lifts       = [result[2][0][3] for result in results]
            return list(zip(lhs, rhs, supports, confidences, lifts))
        
        resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
        
        ## Displaying the results non sorted
        print(resultsinDataFrame)
        
        ## Displaying the results sorted by descending lifts
        print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
        
    def eclat(self,dir):
        
        #Data Preprocessing
        dataset = pd.read_csv(dir, header = None)
        transactions = []
        for i in range(0, 7501):
          transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
        
        # Training the Eclat model on the dataset
        from apyori import apriori
        rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
        
        # Visualising the results
        
        ## Displaying the first results coming directly from the output of the apriori function
        results = list(rules)
        print(results)
        
        ## Putting the results well organised into a Pandas DataFrame
        def inspect(results):
            lhs         = [tuple(result[2][0][0])[0] for result in results]
            rhs         = [tuple(result[2][0][1])[0] for result in results]
            supports    = [result[1] for result in results]
            return list(zip(lhs, rhs, supports))
        resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
        
         ## Displaying the results non sorted
        print(resultsinDataFrame)
        
        ## Displaying the results sorted by descending supports
        print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))
        
class reinforcementLearning:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def upperConfidenceBound(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        
        # Implementing UCB
        import math
        N = 10000
        d = 10
        ads_selected = []
        numbers_of_selections = [0] * d
        sums_of_rewards = [0] * d
        total_reward = 0
        for n in range(0, N):
            ad = 0
            max_upper_bound = 0
            for i in range(0, d):
                if (numbers_of_selections[i] > 0):
                    average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                    delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    ad = i
            ads_selected.append(ad)
            numbers_of_selections[ad] = numbers_of_selections[ad] + 1
            reward = dataset.values[n, ad]
            sums_of_rewards[ad] = sums_of_rewards[ad] + reward
            total_reward = total_reward + reward
        
        # Visualising the results
        plt.hist(ads_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
        
    def thompsonSampling(self,dir):
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        
        # Implementing Thompson Sampling
        import random
        N = 10000
        d = 10
        ads_selected = []
        numbers_of_rewards_1 = [0] * d
        numbers_of_rewards_0 = [0] * d
        total_reward = 0
        for n in range(0, N):
            ad = 0
            max_random = 0
            for i in range(0, d):
                random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
                if random_beta > max_random:
                    max_random = random_beta
                    ad = i
            ads_selected.append(ad)
            reward = dataset.values[n, ad]
            if reward == 1:
                numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
            else:
                numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
            total_reward = total_reward + reward
        
        # Visualising the results - Histogram
        plt.hist(ads_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
        
class naturalLanguageProcessing:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
    
    def bagOfWordsNB(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        # Training the Naive Bayes model on the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
        
    def bagOfWordsLR(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        # Training the Logistic Regression model on the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
    
    def bagOfWordsKNN(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        
        # Training the K-NN model on the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)

        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
    def bagOfWordsSVM(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        
        
        # Training the SVM model on the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)


        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
    def bagOfWordsKSVM(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        
        
               
        # Training the Kernel SVM model on the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)


        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

    def bagOfWordsDTC(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        
        
                
        # Training the Decision Tree Classification model on the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)


        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
    def bagOfWordsRFC(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        
        # Training the Random Forest Classification model on the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)

        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
        
    def bagOfWordsXGB(self,dir):
                
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
          review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
          review = review.lower()
          review = review.split()
          ps = PorterStemmer()
          all_stopwords = stopwords.words('english')
          all_stopwords.remove('not')
          review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
          review = ' '.join(review)
          corpus.append(review)
        print(corpus)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        
        # Training the Random Forest Classification model on the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)

        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
        
class deepLearning:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
    
    def artificialNeuralNetwork(self,dir):
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        tf.__version__
        
        # Part 1 - Data Preprocessing
        
        # Importing the dataset
        dataset = pd.read_csv('Churn_Modelling.csv')
        X = dataset.iloc[:, 3:-1].values
        y = dataset.iloc[:, -1].values
        print(X)
        print(y)
        
        # Encoding categorical data
        # Label Encoding the "Gender" column
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])
        print(X)
        # One Hot Encoding the "Geography" column
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        print(X)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # Part 2 - Building the ANN
        
        # Initializing the ANN
        ann = tf.keras.models.Sequential()
        
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        
        # Adding the second hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        # Part 3 - Training the ANN
        
        # Compiling the ANN
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Training the ANN on the Training set
        ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
        
        # Part 4 - Making the predictions and evaluating the model
        
        # Predicting the result of a single observation
        
        """
        Use our ANN model to predict if the customer with the following informations will leave the bank: 
        Geography: France
        Credit Score: 600
        Gender: Male
        Age: 40 years old
        Tenure: 3 years
        Balance: $ 60000
        Number of Products: 2
        Does this customer have a credit card? Yes
        Is this customer an Active Member: Yes
        Estimated Salary: $ 50000
        So, should we say goodbye to that customer?
        
        Solution:
        """
        
        print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
        
        """
        Therefore, our ANN model predicts that this customer stays in the bank!
        Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
        Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
        """
        
        # Predicting the Test set results
        y_pred = ann.predict(X_test)
        y_pred = (y_pred > 0.5)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)
    def convolutionalNeuralNetwork(self,dir):
        
        # Importing the libraries
        import tensorflow as tf
        from keras.preprocessing.image import ImageDataGenerator
        tf.__version__
        
        # Part 1 - Data Preprocessing
        
        # Preprocessing the Training set
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                         target_size = (64, 64),
                                                         batch_size = 32,
                                                         class_mode = 'binary')
        
        # Preprocessing the Test set
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
        
        # Part 2 - Building the CNN
        
        # Initialising the CNN
        cnn = tf.keras.models.Sequential()
        
        # Step 1 - Convolution
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        
        # Step 2 - Pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        # Adding a second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        # Step 3 - Flattening
        cnn.add(tf.keras.layers.Flatten())
        
        # Step 4 - Full Connection
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        # Step 5 - Output Layer
        cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        # Part 3 - Training the CNN
        
        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Training the CNN on the Training set and evaluating it on the Test set
        cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
        
        # Part 4 - Making a single prediction
        
        import numpy as np
        from keras.preprocessing import image
        test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
            
        print(prediction)