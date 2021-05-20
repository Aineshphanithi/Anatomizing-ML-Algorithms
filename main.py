import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QWidget
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap
import MLModels
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QDialog,QProgressBar, QPushButton)


class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi("login.ui",self)
        self.loginbutton.clicked.connect(self.loginfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password) #To hide the entered password
        self.createaccbutton.clicked.connect(self.gotocreate)
        
    def loginfunction(self):
        email = self.email.text()
        password = self.password.text()
        print("Successfully Logged in with Email: ", email, "and Password:", password)
        selectModelfn = selectModel()
        widget.addWidget(selectModelfn)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def gotocreate(self):
        #Going to the Create Account Widget
        createacc = CreateAcc()
        widget.addWidget(createacc)
        widget.setCurrentIndex(widget.currentIndex()+1) 


class CreateAcc(QDialog):
    def __init__(self):
        super(CreateAcc,self).__init__()
        loadUi("createacc.ui",self)
        self.signupbutton.clicked.connect(self.createaccfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpass.setEchoMode(QtWidgets.QLineEdit.Password)

    def createaccfunction(self):
        email = self.email.text()
        
        if self.password.text() == self.confirmpass.text():
            password = self.password.text()
            print("Successfully created account with Email: ",email, "and Password:",password)
            login=Login()
            widget.addWidget(login)
            widget.setCurrentIndex(widget.currentIndex()+1)

class selectModel(QDialog):
    selectModel=0
    # def __init__(self):
    #      super(selectModel,self).__init__()
    #      loadUi("selectModel.ui",self)
    #      self.regressionbutton.clicked.connect(self.regressionfunction)

    # def regressionfunction(self):
    #     b = regressionUI()
    #     widget.addWidget(b)
    #     widget.setCurrentIndex(widget.currentIndex()+1)
    
    def __init__(self):
        super(selectModel,self).__init__()
        loadUi("selectModel.ui",self)
        self.regressionbutton.clicked.connect(self.regression)
        self.classificationbutton.clicked.connect(self.classification)
        self.clusteringbutton.clicked.connect(self.clustering)
        self.reinforcementlearningbutton.clicked.connect(self.reinforcementLearning)
        self.associationrulelearningbutton.clicked.connect(self.associationRuleLearning)
        self.naturallanguageprocessingbutton.clicked.connect(self.naturalLanguageProcessing)
        self.deeplearningbutton.clicked.connect(self.deepLearning)
        
        
    def regression(self):
        print("regression")
        selectModel=0
        print("selectedModel",selectModel)
        self.callClass(selectModel)

    def classification(self):
        print("classification")
        selectModel=1
        print("selectedModel",selectModel)
        self.callClass(selectModel)

    def clustering(self):
        print("clustering")
        selectModel=2
        print("selectedModel",selectModel)
        self.callClass(selectModel)

    def associationRuleLearning(self):
        print("associationRuleLearning")
        selectModel=3
        print("selectedModel",selectModel)
        self.callClass(selectModel)

    def reinforcementLearning(self):
        print("reinforcementLearning")
        selectModel=4
        print("selectedModel",selectModel)
        self.callClass(selectModel)
    
    def naturalLanguageProcessing(self):
        print("Natural Language Processing")
        selectModel=5
        print("selectedModel",selectModel)
        self.callClass(selectModel)

    def deepLearning(self):
        print("Deep Learning")
        selectModel=6
        print("selectedModel",selectModel)
        self.callDL()
    
    def callDL(self):
        dl = deepLearningUI()
        widget.addWidget(dl)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def callClass(self,selectmodel):
        dataUI = datasetUI(selectmodel)
        widget.addWidget(dataUI)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
class deepLearningUI(QDialog):
    def __init__(self):
        super(deepLearningUI,self).__init__()
        loadUi("deepLearning.ui",self)        
        self.annbutton.clicked.connect(self.browsefiles1)
        self.cnnbutton.clicked.connect(self.browsefiles2)
        self.back.clicked.connect(self.goback)
    
    def goback(self):
        selModScreen = selectModel()
        widget.addWidget(selModScreen)
        widget.setCurrentIndex(widget.currentIndex()+1)  
    
    def browsefiles1(self):
        print("Deep Learning ANN")
        selectModel=6
        print("selectedModel",selectModel)
        self.callClass(selectModel)

    def browsefiles2(self):
        print("Deep Learning CNN")
        selectModel=7
        print("selectedModel",selectModel)
        self.callClass(selectModel)
    
    def callClass(self,test):
        dataUI = datasetUI(test)
        widget.addWidget(dataUI)
        widget.setCurrentIndex(widget.currentIndex()+1)

class datasetUI(QDialog, QThread):
    stri = ""
    path=""
    currModel = 0
    pstr = ""
    def __init__(self,selectmodel):
        
        super(datasetUI,self).__init__()
        print("In constructor ",selectmodel)
        self.currModel = selectmodel
        loadUi("browse.ui",self)        
        self.browsebutton.clicked.connect(self.browsefiles)
        self.progressBar.setValue(0)
        self.back.clicked.connect(self.goback)
        

    def goback(self):

        selModScreen = selectModel()
        widget.addWidget(selModScreen)
        widget.setCurrentIndex(widget.currentIndex()+1)         

    def browsefiles(self,ML):

        if(self.currModel==7):
            fname = QFileDialog.getExistingDirectory(self,'Open Directory','E:\Machine Learning\Machine Learning A-Z (Codes and Datasets)')
            self.path = fname
            self.filename.setText(fname)
            self.stri += "Dataset Loaded from -" + fname
        else:
            fname = QFileDialog.getOpenFileName(self, 'Open File','E:\Machine Learning\Machine Learning A-Z (Codes and Datasets)')
            self.path = fname[0]
            self.filename.setText(fname[0])
            self.stri += "Dataset Loaded from -" + fname[0]
        self.scores.setText(self.stri)
        print(fname)
        self.predictButton.clicked.connect(self.getScores)
        
        self.custompredict.clicked.connect(self.getScores)

        
        #str.isalnum()
    
    def getScores(self):

        self.pstr = self.predicttext.toPlainText()
        print("User Input: ", self.pstr)
        if(',' in self.pstr):    
            self.pstr = self.pstr.split(",")
        temppstr=[]
        for x in self.pstr:
            if(x.isalpha()==False):
                print(x.isalnum(),x)
                temppstr.append(float(x))
            else:
                temppstr.append(x)
        
        self.pstr = [temppstr]

        currModel = self.currModel
        print("Current Model:",currModel)
        scores=[]
        
        classML = [MLModels.regression(),MLModels.classification(),MLModels.clustering(),MLModels.associationRuleLearning(),MLModels.reinforcementLearning(),MLModels.naturalLanguageProcessing(),MLModels.deepLearning(),MLModels.deepLearning()]
        print(classML)
        ML = classML[currModel]
        modules = []
        count=0
        self.stri += "\n Models Initaialized"
        self.scores.setText(self.stri)
        if(currModel==0):
            
            modules.append(ML.multipleLinearRegression(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
            modules.append(ML.polynomialRegression(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
            modules.append(ML.supportVectorRegression(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
            modules.append(ML.decisionTreeRegression(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
            modules.append(ML.randomForestRegression(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
            modules.append(ML.xgBoostR(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
            
            modules.append(ML.catBoostR(self.path,self.pstr))
            count += 100/7
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            
        elif(currModel == 1):
            
            modules.append(ML.logisticRegression(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.kNearestNeighbors(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.supportVectorMachine(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.kernelSupportVectorMachine(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.naiveBayes(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.decisionTreeClassification(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.randomForestClassification(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.xgBoostC(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           
            modules.append(ML.catBoostC(self.path,self.pstr))
            count += 100/9
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
           

        elif(currModel == 2):
            
            modules.append(ML.k_MeansClustering(self.path,self.pstr))
            count += 100/2
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            modules.append(ML.hierarchicalClustering(self.path,self.pstr))
            count += 100/2
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
        
        elif(currModel == 3):
            
            modules.append(ML.apriori(self.path))
            count += 100/2
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            modules.append(ML.eclat(self.path))
            count += 100/2
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

        elif(currModel == 4):

            modules.append(ML.upperConfidenceBound(self.path))
            count += 100/2
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.thompsonSampling(self.path))
            count += 100/2
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

        
        elif(currModel == 5):

            modules.append(ML.bagOfWordsNB(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsLR(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsKNN(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsSVM(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsKSVM(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsDTC(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsRFC(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

            modules.append(ML.bagOfWordsXGB(self.path))
            count += 100/8
            print("Progress Bar is updating")
            self.progressBar.setValue(count)


        elif(currModel == 6):

            modules.append(ML.artificialNeuralNetwork(self.path))
            count += 100/3
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            modules.append(ML.artificialNeuralNetwork2(self.path))
            count += 100/3
            print("Progress Bar is updating")
            self.progressBar.setValue(count)
            modules.append(ML.artificialNeuralNetwork3(self.path))
            count += 100/3
            print("Progress Bar is updating")
            self.progressBar.setValue(count)

        elif(currModel == 7):

            modules.append(ML.convolutionalNeuralNetwork(self.path))
            self.progressBar.setValue(100)
        
        else:
        
            print("error: Selected Model Doesn't exist")
        #for func in modules[selectedModel]:

        print(modules,currModel)
        
        for func in modules:
            
            if(currModel == 0):
                score = func
                #print(score)
                
                scores.append(score[1].real)
                scoresstr = [str(x) for x in scores]
                if(score[-1][0]!=-999):
                    self.stri+="\n RMSE: "+" ".join(scoresstr)+"\nPrediction Results : "+str(score[-1][0])
                else:
                    self.stri+="\n RMSE: "+" ".join(scoresstr)
                self.scores.setText(self.stri)
                
            elif(currModel == 1):
                score = func
                print(score)
                scores.append(score[0])
                scoresstr = [str(x) for x in scores]
                if(score[1][0]!=-999):
                    self.stri+="\n RMSE: "+" ".join(scoresstr)+"\nPrediction Results : "+str(score[1][0])
                else:
                    self.stri+="\n RMSE: "+" ".join(scoresstr)
                self.scores.setText(self.stri)

            elif(currModel == 2):
                score = func
                print(score)
                
                scores.append(score[0])
                scoresstr = [str(x) for x in scores]
                if(score[1][0]!=-999):
                    self.stri+="\n RMSE: "+" ".join(scoresstr)+"\nPrediction Results : "+str(score[1][0])
                else:
                    self.stri+="\n RMSE: "+" ".join(scoresstr)
                self.scores.setText(self.stri)

            elif(currModel == 3):
                score = func
                print(score)
                scores.append(score)
                scorestr = [str(x) for x in scores]
                self.scores.setText(self.stri+"\n"+'\n'.join(scorestr))
            
            elif(currModel == 4):
                score = func
                print(score)
                scores.append(score)
                scorestr = [str(x) for x in scores]
                self.scores.setText(self.stri+"\n"+'\n'.join(scorestr))

            elif(currModel == 5):
                score = func
                print(score)
                scores.append(score)
                scoresstr = [str(x) for x in scores]
                self.scores.setText(self.stri+"\n"+" ".join(scoresstr))

            elif(currModel == 6):
                score = func
                print(score)
                scores.append(score[0])
                scoresstr = [str(x) for x in scores]
                self.scores.setText(self.stri+"\n"+"\n ".join(scoresstr))

            elif(currModel == 7):
                score = func
                print(score)
                scores.append(score)
                scoresstr = [str(x) for x in scores]
                self.scores.setText(self.stri+"\n"+" ".join(scoresstr))

        import matplotlib.pyplot as plt; plt.rcdefaults()
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.cluster.hierarchy as sch
        #from mpl_toolkits.axes_grid1 import make_axes_locatable

        
        print("selectedModel",currModel)
        if(currModel==0):
            objects = ('multipleLinearRegression','polynomialRegression','supportVectorRegression','decisionTreeRegression','randomForestRegression','xgBoostR','catBoostR')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.xticks(rotation=25)
            plt.ylabel('RMSE')
            plt.title('Regression Models RMSE value')

        elif(currModel == 1):
            objects = ('LogisticRegression','KNearestNeighbors','SupportVectorMachine','KernelSupportVectorMachine','NaiveBayes','DecisionTreeClassification','RandomForestClassification','xgBoostC','CatBoostC')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.xticks(rotation=25)
            plt.ylabel('Accuracy Score')
            plt.title('Classification Models Accuracy Scores')

        elif(currModel == 2):
            objects = ('K_Means Clustering','Hierarchical Clustering')
            y_pos = np.arange(len(objects))
            #performance = scores
            print("In plot : ")
            
            kmeans = modules[0][4]
            X = modules[0][3]
            y_kmeans = modules[0][2]
            X2 = modules[1][2]
            wcss = modules[0][5]
            
            print("In plot : ",y_kmeans,X)
            
            fig, axes = plt.subplots(2,2)
            
            ax1 = axes[0][0]
            ax2 = axes[0][1]
            ax3 = axes[1][0]
            ax4 = axes[1][1]

            ax3.plot(range(1, 11), wcss)
            ax3.set_title('The Elbow Method')
            
            dendrogram = sch.dendrogram(sch.linkage(X2, method = 'ward'))
            ax4.set_title('Dendrogram')
            ax4.plot()

            ax1.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')#plotting the first cluster's point
            ax1.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
            ax1.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
            ax1.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
            ax1.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
            #ax3.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'centroids')
            ax1.set_title('K_Means Clustering')
            
            X=X2
            y_hc= modules[1][3]
            ax2.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
            ax2.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
            ax2.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
            ax2.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
            ax2.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
            ax2.set_title('Hierarchical_clustering')

        elif(currModel == 3):
            df=modules[0]
            # lhs = df['Left Hand Side'].tolist()
            # rhs = df['Right Hand Side'].tolist()
            # sup = df['Support'].tolist()
            # con = df['Confidence'].tolist()
            # lif = df['Lift'].tolist()
            # data = df.values.tolist()
            # column_labels = ["Left Hand Side","Right Hand Side","Support","Confidence","Lift"]

            fig, ax = plt.subplots(2,1)
            print(ax)
            ax1=ax[0]
            ax2=ax[1]
            
            #ax1.axis('tight')
            ax1.axis('off')
            ax1.set_title('Apriori (Association Rule Learning)')
            table = ax1.table(cellText=df.values,colLabels=df.columns,colColours = ["blue"]*len(df.columns),loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            
            df = modules[1]
            #ax2.axis('tight')
            ax2.axis('off')
            ax2.set_title('Eclat (Association Rule Learning)')
            table = ax2.table(cellText=df.values,colLabels=df.columns,colColours = ["blue"]*len(df.columns),loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)

        elif(currModel == 4):
            fig, ax = plt.subplots(2,1)
            print(ax)
            ax1=ax[0]
            ax2=ax[1]
            # Visualising the results
            ads_selected = modules[0]
            ax1.hist(ads_selected)
            ax1.set_title('Histogram of Selections(Upper Confidence Bound)')
            # plt.xlabel('Ads')
            # plt.ylabel('Number of times each ad was selected')
            # plt.title('Reinforcement Learning Model')
            ads_selected = modules[1]
            ax2.hist(ads_selected)
            ax2.set_title('Histogram of Selections(Thompson Sampling)')
            
        elif(currModel == 5):
            objects = ('bagOfWordsNB','bagOfWordsLR','bagOfWordsKNN','bagOfWordsSVM','bagOfWordsKSVM','bagOfWordsDTC','bagOfWordsRFC','bagOfWordsXGB')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.xticks(rotation=25)
            plt.ylabel('Accuracy')
            plt.title('Natural Language Processing Using Classification')

        elif(currModel == 6):
            print("ANN")
            
            # summarize history for accuracy
            fig, ax = plt.subplots(3,3)
            print(ax)
            
            ax1=ax[0][0]
            ax2=ax[1][0]
            ax3=ax[2][0]
            ax4=ax[0][1]
            ax5=ax[1][1]
            ax6=ax[2][1]
            ax7=ax[0][2]
            ax8=ax[1][2]
            ax9=ax[2][2]
            
            hist = modules[0][2]
            ax1.plot(hist.history['accuracy'])
            ax1.plot(hist.history['val_accuracy'])
            ax1.set_title('model accuracy (2-Layers)')
            ax1.legend(['train', 'test'], loc='upper left')
            
            # summarize history for loss
            ax2.plot(hist.history['loss'])
            ax2.plot(hist.history['val_loss'])
            ax2.set_title('model loss')
            ax2.legend(['train', 'test'], loc='upper left')
            

            cm = modules[0][1]
            ax3.set_title('Confusion Matrix')
            ax3.matshow(cm,cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax3.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')


            hist = modules[1][2]
            ax4.plot(hist.history['accuracy'])
            ax4.plot(hist.history['val_accuracy'])
            ax4.set_title('model accuracy (5-Layers)')
            ax4.legend(['train', 'test'], loc='upper left')
            
            # summarize history for loss
            ax5.plot(hist.history['loss'])
            ax5.plot(hist.history['val_loss'])
            ax5.set_title('model loss')
            ax5.legend(['train', 'test'], loc='upper left')
            

            cm = modules[1][1]
            ax6.set_title('Confusion Matrix')
            ax6.matshow(cm,cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax6.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')

            hist = modules[2][2]
            ax7.plot(hist.history['accuracy'])
            ax7.plot(hist.history['val_accuracy'])
            ax7.set_title('model accuracy (10-Layers)')
            ax7.legend(['train', 'test'], loc='upper left')
            
            # summarize history for loss
            ax8.plot(hist.history['loss'])
            ax8.plot(hist.history['val_loss'])
            ax8.set_title('model loss')
            ax8.legend(['train', 'test'], loc='upper left')
            

            cm = modules[2][1]
            ax9.set_title('Confusion Matrix')
            ax9.matshow(cm,cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax9.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')

            plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

            
        elif(currModel == 7):
            print("CNN")
            hist = modules[0][1]
            plt.plot(hist.history['accuracy'])
            plt.plot(hist.history['val_accuracy'])
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('CNN Traning vs Validation Scores')
            plt.legend(['train_accuracy', 'val_accuracy','train__loss', 'val_loss'], loc='upper left')
            

        plt.show()
        print(scores)


app = QApplication(sys.argv)
mainWindow = selectModel()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedWidth(480)
widget.setFixedHeight(620)
widget.show()
app.exec_()