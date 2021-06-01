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
        
    def multipleLinearRegression(self,dir,pstr,select):
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
        print("pstr :",pstr)
        if(len(pstr[0])!=0):
            print("In merging")
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            
            #pstr = np.array(ct.fit_transform(pstr))
        if(len(pstr[0])!=0):
            print("In extraction")
            pstr = np.array([X[-1]])
            X = X[0:-1]

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
        if(len(pstr[0])!=0):
            print("In prediction custom")
            cstpred = regressor.predict(pstr)
        else:
            print("In prediction null")
            cstpred = [-999]
        #accScore = accuracy_score(y_test, y_pred)
        np.set_printoptions(precision=2)
        
        print(X_test,pstr)
        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("multiple Linear Regression")
        
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse,cstpred]

    def polynomialRegression(self,dir,pstr,select):
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
        
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            #pstr = np.array(ct.fit_transform(pstr))
            #print(X)
        #print(y)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]


        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        #print(y)
        
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
        if(len(pstr[0])!=0):
            cstpred = lin_reg_2.predict(poly_reg.fit_transform(pstr))
        else:
            cstpred = [-999]

        np.set_printoptions(precision=2)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
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
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse,cstpred]
        
    def supportVectorRegression(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        #print(X)
        #print(y)
        y = y.reshape(len(y),1)
        #print(y)
         # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))

        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            #pstr = np.array(ct.fit_transform(pstr))
            #print(X)
        #print(y)
        

        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        #print(y)
        # Splitting the dataset into the Training set and Test set
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)
        #print(X)
        #print(y)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]

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
        if(len(pstr[0])!=0):
            cstpred = sc_y.inverse_transform(regressor.predict(pstr))
        else:
            cstpred = [-999]
        np.set_printoptions(precision=2)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        #graph of pred vs test
        
        # plt.plot(y_test)
        # plt.plot(y_pred)
        # plt.legend(["y_test","y_pred"])
        # plt.title('predicted vs actual result')
        # plt.xlabel("Years of Experience")
        # plt.ylabel('Salary')
        # plt.savefig('regression.jpg',bbox_inches = 'tight', dpi = 150 )

        #plt.show()
        #print(y_test,y_pred)

        from sklearn.metrics import mean_squared_error
        import cmath
        mse = mean_squared_error(y_test, y_pred)
        rmse = cmath.sqrt(mse)
        print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse,cstpred]

    def decisionTreeRegression(self,dir,pstr,select):
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
        
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            #pstr = np.array(ct.fit_transform(pstr))
            #print(X)
        #print(y)
        
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        #print(y)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
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

        if(len(pstr[0])!=0):
            cstpred = regressor.predict(pstr)
        else:
            cstpred = [-999]
        np.set_printoptions(precision=2)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
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
        #print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse,cstpred]
        
    def randomForestRegression(self,dir,pstr,select):
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
        
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            
            #print(X)
        #print(y)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        #print(y)
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
        if(len(pstr[0])!=0):
            cstpred = regressor.predict(pstr)
        else:
            cstpred = [-999]

        np.set_printoptions(precision=2)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
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
        #print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse,cstpred]

    def xgBoostR(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        #https://www.geeksforgeeks.org/xgboost-for-regression/
        # Necessary imports
        import xgboost as xg
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error as MSE
        
        #print("Load the data")
        dataset = pd.read_csv(dir)
        X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
          
        #print("Encoding categorical data")
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
       # print("removing the string from independent variable")
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            
            
        #print(y)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]

        from sklearn.preprocessing import LabelEncoder
        if isinstance(y[0], str):
            #print("removing the string from dependent variable")
            le = LabelEncoder()
            y = le.fit_transform(y)
        #print(y)

        print("XGBoost")
       # print("Splitting")
        train_X, test_X, train_y, y_test = train_test_split(X, y,test_size = 0.3, random_state = 123)
          
        #print("Instantiation")
        xgb_r = xg.XGBRegressor(objective ='reg:linear',
                          n_estimators = 10, seed = 123)
          
        # Fitting the model
        xgb_r.fit(train_X, train_y)
        
        # Predict the model
        y_pred = xgb_r.predict(test_X)
        print(type(test_X),type(pstr))
        
        if(len(pstr[0])!=0):
            cstpred = xgb_r.predict(np.array(pstr))
        else:
            cstpred = [-999]
        #print("xgboost Regression")
        # RMSE Computation
        rmse = np.sqrt(MSE(y_test, y_pred))
        #print("RMSE : % f" %(rmse))

        np.set_printoptions(precision=2)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
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
        #print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse,cstpred]

    def catBoostR(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        #https://catboost.ai/docs/concepts/python-usages-examples.html
        from catboost import CatBoostRegressor
        
        # Importing the dataset
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        #print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            
            #print(X)
        
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        #print(y)
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
        if(len(pstr[0])!=0):
            cstpred = model.predict(pstr)
        else:
            cstpred = [-999]
        np.set_printoptions(precision=2)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
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
        #print("polynomial Regression")
        return [str(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)),rmse, cstpred]

    def graphs(self):
        #Under Construction
        print("hello")

class classification:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def logisticRegression(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
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
        
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            # print(X)
        
        

        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(y[0], str):
            # removing the string from dependent variable
            le = LabelEncoder()
            y = le.fit_transform(y)
        # print(y)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
  
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # print(X_train)
        # print(X_test)
        
        # Training the Logistic Regression model on the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('logisticregression.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def kNearestNeighbors(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
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
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))

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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]

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
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('knn.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]

        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score=accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def supportVectorMachine(self,dir,pstr,select):
                
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
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
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('supportvectormachine.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
                print(classifier)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def kernelSupportVectorMachine(self,dir,pstr,select):
                
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))

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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]

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
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('kernelsupportvectormachine.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def naiveBayes(self,dir,pstr,select):
 
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
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
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('naivebayes.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def decisionTreeClassification(self,dir,pstr,select):
        
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
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
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('descisiontreeclassification.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def randomForestClassification(self,dir,pstr,select):
                
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
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
        if(select==0):    
            classifier.fit(X_train, y_train)
        else:
            with open('randomforestclassification.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        
        # Predicting a new result
        print(classifier.predict(sc.transform([[30,87000]])))
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        return [score,cstpred]

    def xgBoostC(self,dir,pstr,select):
        
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Training XGBoost on the Training set
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        if(select==0):    
            classifier.fit(X_train, y_train)
            # with open('xgboost.pkl', 'wb') as file:
                
            #     # A new file will be created
            #     pickle.dump(classifier, file)
        else:
            with open('xgboost.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred = classifier.predict(X_test)
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        
        # Applying k-Fold Cross Validation
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
        return [score,cstpred]
        
    def catBoostC(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(X[0],y[0])
        # Encoding categorical data
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
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
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        from catboost import CatBoostClassifier
        
        classifier = CatBoostClassifier()
        if(select==0):    
            classifier.fit(X_train, y_train)
            # with open('catboost.pkl', 'wb') as file:
                
            #     # A new file will be created
            #     pickle.dump(classifier, file)
        else:
            with open('catboost.pkl', 'rb') as file:
      
                # Call load method to deserialze
                classifier = pickle.load(file)
            
                print(classifier)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        
        y_pred = classifier.predict(X_test)
        if(len(pstr[0])!=0):
            cstpred = classifier.predict(pstr)
        else:
            cstpred = [-999]
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_test, y_pred)
        
        # Applying k-Fold Cross Validation
        from sklearn.model_selection import cross_val_score
        
        # accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
        # print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
        # print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
        return [score,cstpred]

class clustering:
    
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def k_MeansClustering(self,dir,pstr,select):
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import v_measure_score
        #from sklearn.metrics import silhouette_score
        
        dataset = pd.read_csv(dir)
        X = dataset.iloc[:,[3,4]].values
        #y = dataset.iloc[:, -1].values
        pstr = pd.DataFrame(pstr)
        pstr = pstr.iloc[:].values
        
        pstr = pstr[:,[3,4]]
        pstr=np.array(pstr)
        
        
        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,pstr))
        print(X)
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
        
        # from sklearn.preprocessing import LabelEncoder
        # if isinstance(y[0], str):
        #     # removing the string from dependent variable
        #     le = LabelEncoder()
        #     y = le.fit_transform(y)
        # Using the elbow method to find the optimal number of clusters
        
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X = X[0:-1]
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # V Measure
        v_scores = []
        # print(X[0],X[-1])
        # for i in range(1,11):
        #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        #     kmeans.fit(X)
        #     y_pred = kmeans.fit_predict(X)
        #     v_scores.append(v_measure_score(y, y_pred))

        # plt.plot(range(1, 11), wcss)
        # plt.title('The Elbow Method')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('WCSS')
        # plt.show()
        # bestC = v_scores.index(max(v_scores))
        # plt.plot(range(1, 11), v_scores)
        # plt.title('V Measure')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('score')
        #plt.show()
        
        # Training the K-Means model on the dataset
        kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
        y_pred = kmeans.fit_predict(X)
        print(X)
        
        print(pstr)
        pstr = np.array(pstr)
        print(pstr)
        if(len(pstr[0])!=0):
            cstpred = kmeans.fit_predict(pstr)
        else:
            cstpred = [-999]
        print(y_pred)
        y_kmeans = kmeans.fit_predict(X)
        
        # plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')#plotting the first cluster's point
        # plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
        # plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
        # plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
        # plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'centroids')
        # plt.title('Clusters of customers')
        # plt.xlabel('Annual Income (K$)')
        # plt.ylabel('Spending Score (1-100)')
        # plt.legend()
        #plt.subplot(1,2,1)

        return [v_scores,cstpred,y_kmeans,X,kmeans,wcss]

    def hierarchicalClustering(self,dir,pstr,select):
        
        # Importing the dataset
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import v_measure_score

        dataset = pd.read_csv(dir)
        X = dataset.iloc[:,[3,4]].values
        #y = dataset.iloc[:, -1].values
        
        # X = dataset.iloc[:, :-1].values
        # y = dataset.iloc[:, -1].values

        stridx = []
        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
        if(len(pstr[0])!=0):
            X = np.vstack((X,np.array(pstr[0])))
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            print(X)
        
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X = X[0:-1]
        # from sklearn.preprocessing import LabelEncoder
        # if isinstance(y[0], str):
        #     # removing the string from dependent variable
        #     le = LabelEncoder()
        #     y = le.fit_transform(y)
        
        
        #WARNING for huge data it will get error
        # Using the dendrogram to find the optimal number of clusters
        # import scipy.cluster.hierarchy as sch
        # dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        # plt.title('Dendrogram')
        # plt.xlabel('Customers')
        # plt.ylabel('Euclidean distances')
        #plt.subplot(1,2,2)
        #plt.plot()
        # Training the Hierarchical Clustering model on the dataset
        from sklearn.cluster import AgglomerativeClustering
        v_scores = []
        # for i in range(1,11):
        #     hc = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'ward')
        #     y_pred = hc.fit_predict(X)
           
        #     v_scores.append(v_measure_score(y, y_pred))
        
        # bestC = v_scores.index(max(v_scores))
        # plt.plot(range(1, 11), v_scores)
        # plt.title('V Measure')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('score')
        # plt.show()
       
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_pred = hc.fit_predict(X)
        y_hc = y_pred
        if(len(pstr[0])!=0):
            cstpred = hc.fit_predict(pstr)
        else:
            cstpred = [-999]
        return [v_scores,cstpred,X,y_hc]

class associationRuleLearning:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def apriori(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import pickle
        # Data Preprocessing
        dataset = pd.read_csv(dir, header = None)
        transactions = []
        for i in range(0, 7501):
          transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
        
        # Training the Apriori model on the dataset
        from apyori import apriori
        if(select==0):
            rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
        else:
            with open('apriori.pkl', 'rb') as file:
      
                # Call load method to deserialze
                rules = pickle.load(file)
  
            print(rules)
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
        if(len(pstr[0]) == 0):
            prdres1 = -999
            prdres2 = -999
        else:
            prdres1 = resultsinDataFrame['Right Hand Side'].where(resultsinDataFrame['Left Hand Side'] == pstr[0][0])
            prdres2 = "Lift: \n"+str(resultsinDataFrame['Lift'].where(resultsinDataFrame['Left Hand Side'] == pstr[0][0]))

        ## Displaying the results non sorted
        print(resultsinDataFrame)
        
        ## Displaying the results sorted by descending lifts
        result = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
        print(result)
        #print(pstr[0][0],resultsinDataFrame['Right Hand Side'].where(resultsinDataFrame['Left Hand Side'] == 'pasta'))
        if(len(pstr[0]) == 0):
            prdres1 = -999
            prdres2 = -999
        else:
            prdres1 = resultsinDataFrame['Right Hand Side'].where(resultsinDataFrame['Left Hand Side'] == pstr[0][0])
            prdres2 = "Lift: \n"+str(resultsinDataFrame['Lift'].where(resultsinDataFrame['Left Hand Side'] == pstr[0][0]))

        return [result,prdres1,prdres2]

    def eclat(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
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
        result = resultsinDataFrame.nlargest(n = 10, columns = 'Support')
        print(result)
        if(len(pstr[0]) == 0):
            prdres1 = -999
            prdres2 = -999
        else:
            prdres1 = resultsinDataFrame['Product 2'].where(resultsinDataFrame['Product 1'] == pstr[0][0])
            prdres2 = "Support : \n"+str(resultsinDataFrame['Support'].where(resultsinDataFrame['Product 1'] == pstr[0][0]))

        return [result,prdres1,prdres2]

class reinforcementLearning:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
    def upperConfidenceBound(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir)
        temp=dataset
        X=dataset.iloc[:,].values
        print(len(X),len(X[0]))
        stridx = []

        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
                print("Encoding Here",X[0][x])
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            
        
        # Implementing UCB
        import math
        N = len(X)
        d = len(X[0])
        
        print(X, N, d)
        dataset = pd.DataFrame(X)
        print(dataset)
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
        # plt.hist(ads_selected)
        # plt.title('Histogram of ads selections')
        # plt.xlabel('Ads')
        # plt.ylabel('Number of times each ad was selected')
        #plt.show()
        print(ads_selected)
        return ads_selected
        
    def thompsonSampling(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        # Importing the dataset
        dataset = pd.read_csv(dir)
        temp=dataset
        X=dataset.iloc[:,].values
        print(len(X),len(X[0]))
        stridx = []

        for x in range(0, len(X[0])):
            if isinstance(X[0][x], str):
                stridx.append(x)
                print("Encoding Here",X[0][x])
        
        # removing the string from independent variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        for i in stridx:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            
        
        # Implementing UCB
        import math
        N = len(X)
        d = len(X[0])
        
        print(X, N, d)
        dataset = pd.DataFrame(X)
        print(dataset)
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
        # plt.hist(ads_selected)
        # plt.title('Histogram of ads selections')
        # plt.xlabel('Ads')
        # plt.ylabel('Number of times each ad was selected')
        #plt.show()
        print(ads_selected)
        return ads_selected

class naturalLanguageProcessing:
    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
    
    def bagOfWordsNB(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the dataset
        dataset = pd.read_csv(dir, delimiter = '\t', quoting = 3)
        #print(type(dataset))
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
        
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            #print(corpus,prediction)
            corpus.extend(prediction)
        print(corpus)
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
            #pstr = cv.fit_transform(prediction).toarray()
        y = dataset.iloc[:, -1].values
        print(X,pstr)
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
        # Training the Naive Bayes model on the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        print(type(X_test),X_test)
        print(type(pstr),pstr)
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]
          
    def bagOfWordsLR(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd  
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            corpus.extend(prediction)
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
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
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]
    
    def bagOfWordsKNN(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd     
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            # Creating the Bag of Words model
            corpus.extend(prediction)
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
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
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]
        
    def bagOfWordsSVM(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd        
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            corpus.extend(prediction)
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
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
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]
        
    def bagOfWordsKSVM(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd        
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            corpus.extend(prediction)
        # Creating the Bag of Words model

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
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
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]

    def bagOfWordsDTC(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd        
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            corpus.extend(prediction)
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
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
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]
        
    def bagOfWordsRFC(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd        
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            corpus.extend(prediction)
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
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
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]
        
    def bagOfWordsXGB(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd        
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
        if(len(pstr[0])!=0):
            prediction = []
            review = re.sub('[^a-zA-Z]', ' ', pstr[0][0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            prediction.append(review)
            print(corpus,prediction)
            corpus.extend(prediction)
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        if(len(pstr[0])!=0):
            pstr = [X[-1]]
            X=X[0:-1]
        
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
        if(len(pstr[0])!=0):
            pstr = classifier.predict(pstr)
        else:
            pstr = [-999]
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return [accuracy_score(y_test, y_pred),pstr]

class deepLearning:

    def __init__(self):
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
    
    def artificialNeuralNetwork(self,dir,pstr,select):
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        import pickle
        tf.__version__
        

        dataset = pd.read_csv(dir)
        if(len(pstr[0])!=0):   
            pstr[0].append(0)
            dataset.loc[len(dataset)] = pstr[0]
        X = dataset.iloc[:,:-1].values
        
        print("first")
        print(X)
        print(pstr)
        
        print("second")
        print(X)
        
        print(pstr)
        X = X[:, 3:-1]
        if(len(pstr[0])!=0):
            y = dataset.iloc[:-1, -1].values
        else:
            y = dataset.iloc[:,-1].values
        print("third")
        print(X)
      
        print(pstr)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

        print(X,pstr)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        print("fourth")
        print(X)
      
        print(pstr)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        if(len(pstr[0])!=0):
            pstr = sc.transform(pstr)
        print(X,pstr)
        # Part 2 - Building the ANN
        
        # Initializing the ANN
        ann = tf.keras.models.Sequential()
        
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        
        
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        # Part 3 - Training the ANN
        
        # Compiling the ANN
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Training the ANN on the Training set
        
        
        if(select == 0):
            hist = ann.fit(X_train, y_train, validation_split=0.33,batch_size = 32, epochs = 100)
            ann.save_weights('./checkpoints1/my_checkpoint')
            with open('/trainHistoryann1Dict', 'wb') as file_pi:
               pickle.dump(hist.history, file_pi)
        else:
            #ann(np.zeros((1,w,h,c)))
            ann.load_weights('./checkpoints1/my_checkpoint')
            hist = pickle.load(open('/trainHistoryann1Dict', "rb"))


        y_pred = ann.predict(X_test)
        
        y_pred = (y_pred > 0.5)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        acc=accuracy_score(y_test, y_pred)
        if(len(pstr[0])!=0):
            pstr = ann.predict(pstr)
        else:
            pstr = -999
        print(pstr)
        return [acc,cm,hist,pstr]

    def artificialNeuralNetwork2(self,dir,pstr,select):
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        import pickle
        tf.__version__
        

        dataset = pd.read_csv(dir)
        if(len(pstr[0])!=0):   
            #pstr[0].append(0)
            print(dataset,pstr)
            dataset.loc[len(dataset)] = pstr[0]

        X = dataset.iloc[:,:-1].values
        
        print("first")
        print(X)
        print(pstr)
        
        print("second")
        print(X)
        
        print(pstr)
        X = X[:, 3:-1]
        if(len(pstr[0])!=0):
            y = dataset.iloc[:-1, -1].values
        else:
            y = dataset.iloc[:,-1].values
        

        print("third")
        print(X)
      
        print(pstr)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

        print(X,pstr)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        print("fourth")
        print(X)
      
        print(pstr)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        if(len(pstr[0])!=0):
            pstr = sc.transform(pstr)
        print(X,pstr)
        # Part 2 - Building the ANN
        
        # Initializing the ANN
        ann = tf.keras.models.Sequential()
        
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=24, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=48, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=98, activation='relu'))
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        # Part 3 - Training the ANN
        
        # Compiling the ANN
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Training the ANN on the Training set
        if(select == 0):
            hist = ann.fit(X_train, y_train, validation_split=0.33,batch_size = 32, epochs = 100)
            ann.save_weights('./checkpoints5/my_checkpoint')
            with open('/trainHistoryann5Dict', 'wb') as file_pi:
               pickle.dump(hist.history, file_pi)
        else:
            
            ann.load_weights('./checkpoints5/my_checkpoint')
            hist = pickle.load(open('/trainHistoryann5Dict', "rb"))

        y_pred = ann.predict(X_test)
        
        y_pred = (y_pred > 0.5)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        acc=accuracy_score(y_test, y_pred)
        if(len(pstr[0])!=0):
            pstr = ann.predict(pstr)
        else:
            pstr = -999
        print(pstr)
        return [acc,cm,hist,pstr]

    def artificialNeuralNetwork3(self,dir,pstr,select):
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        import pickle
        tf.__version__
        

        dataset = pd.read_csv(dir)
        if(len(pstr[0])!=0):   
            #pstr[0].append(0)
            dataset.loc[len(dataset)] = pstr[0]
        X = dataset.iloc[:,:-1].values
        
        print("first")
        print(X)
        print(pstr)
        
        print("second")
        print(X)
        
        print(pstr)
        X = X[:, 3:-1]

        if(len(pstr[0])!=0):
            y = dataset.iloc[:-1, -1].values
        else:
            y = dataset.iloc[:,-1].values

        print("third")
        print(X)
      
        print(pstr)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

        print(X,pstr)
        if(len(pstr[0])!=0):
            pstr = np.array([X[-1]])
            X = X[0:-1]
        print("fourth")
        print(X)
      
        print(pstr)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        if(len(pstr[0])!=0):
            pstr = sc.transform(pstr)
        print(X,pstr)
        # Part 2 - Building the ANN
        
        # Initializing the ANN
        ann = tf.keras.models.Sequential()
        
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=24, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=48, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=98, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=196, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=392, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=784, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1568, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=3136, activation='relu'))
        
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        # Part 3 - Training the ANN
        
        # Compiling the ANN
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Training the ANN on the Training set
        if(select == 0):
            hist = ann.fit(X_train, y_train, validation_split=0.33,batch_size = 32, epochs = 100)
            ann.save_weights('./checkpoints10/my_checkpoint')
            with open('/trainHistoryann10Dict', 'wb') as file_pi:
               pickle.dump(hist.history, file_pi)
        else:
            
            ann.load_weights('./checkpoints10/my_checkpoint')
            hist = pickle.load(open('/trainHistoryann10Dict', "rb"))

        y_pred = ann.predict(X_test)
        
        y_pred = (y_pred > 0.5)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        acc=accuracy_score(y_test, y_pred)
        if(len(pstr[0])!=0):
            pstr = ann.predict(pstr)
        else:
            pstr = -999
        print(pstr)
        return [acc,cm,hist,pstr]

    def convolutionalNeuralNetwork(self,dir,pstr,select):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        # Importing the libraries
        import tensorflow as tf
        from keras.preprocessing.image import ImageDataGenerator
        import pickle
        tf.__version__
        
        # Part 1 - Data Preprocessing
        
        # Preprocessing the Training set
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        training_set = train_datagen.flow_from_directory(dir+'/training_set',
                                                         target_size = (64, 64),
                                                         batch_size = 32,
                                                         class_mode = 'binary')
        
        # Preprocessing the Test set
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_set = test_datagen.flow_from_directory(dir+'/test_set',
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
        if(select == 0):
            hist = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
        else:
            #cnn.save_weights(filepath='final_weight.h5')
            cnn.load_weights('final_weight.h5')
            #with open('/trainHistoryDict', 'wb') as file_pi:
            #    pickle.dump(hist.history, file_pi)
            hist = pickle.load(open('/trainHistoryDict', "rb"))
        
        # Part 4 - Making a single prediction
        if(len(pstr[0])!=0):
        
            import numpy as np
            from keras.preprocessing import image
            test_image = image.load_img(pstr[0][0], target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = cnn.predict(test_image)
            training_set.class_indices
            if result[0][0] == 1:
                prediction = '1st Class'
            else:
                prediction = '2nd Class'
        else:
            prediction = -999
        print(prediction)

        return [hist['accuracy'][-1],hist,prediction]