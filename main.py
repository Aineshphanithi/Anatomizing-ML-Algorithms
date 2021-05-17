import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QWidget
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap
import MLModels
import threading

#selectmodel=0

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
        self.reinforcementlearningbutton.clicked.connect(self.associationRuleLearning)
        self.associationrulelearningbutton.clicked.connect(self.reinforcementLearning)
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

class datasetUI(QDialog):
    path=""
    currModel = 0
    def __init__(self,selectmodel):
        
        super(datasetUI,self).__init__()
        print("In constructor ",selectmodel)
        self.currModel = selectmodel
        loadUi("browse.ui",self)        
        self.browsebutton.clicked.connect(self.browsefiles)
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
        
        else:
            fname = QFileDialog.getOpenFileName(self, 'Open File','E:\Machine Learning\Machine Learning A-Z (Codes and Datasets)')
            self.path = fname[0]
            self.filename.setText(fname[0])
        
        print(fname)
        self.predictButton.clicked.connect(self.getScores)
    
    def getScores(self):
        currModel = self.currModel
        print("Current Model:",currModel)
        scores=[]
        classML = [MLModels.regression(),MLModels.classification(),MLModels.clustering(),MLModels.associationRuleLearning(),MLModels.reinforcementLearning(),MLModels.naturalLanguageProcessing(),MLModels.deepLearning(),MLModels.deepLearning()]
        print(classML)
        ML = classML[currModel]
        if(currModel==0):
            modules = [ML.multipleLinearRegression(self.path),ML.polynomialRegression(self.path),ML.supportVectorRegression(self.path),ML.decisionTreeRegression(self.path),ML.randomForestRegression(self.path),ML.xgBoostR(self.path),ML.catBoostR(self.path)]
        elif(currModel == 1):
            modules = [ML.logisticRegression(self.path),ML.kNearestNeighbors(self.path),ML.supportVectorMachine(self.path),ML.kernelSupportVectorMachine(self.path),ML.naiveBayes(self.path),ML.decisionTreeClassification(self.path),ML.randomForestClassification(self.path),ML.xgBoostC(self.path),ML.catBoostC(self.path)]
        elif(currModel == 2):
            modules = [ML.k_MeansClustering(self.path),ML.hierarchicalClustering(self.path)]
        elif(currModel == 3):
            modules = [ML.apriori(self.path),ML.eclat(self.path)]
        elif(currModel == 4):
            modules = [ML.upperConfidenceBound(self.path),ML.thompsonSampling(self.path)]
        elif(currModel == 5):
            modules = [ML.bagOfWordsNB(self.path),ML.bagOfWordsLR(self.path),ML.bagOfWordsKNN(self.path),ML.bagOfWordsSVM(self.path),ML.bagOfWordsKSVM(self.path),ML.bagOfWordsDTC(self.path),ML.bagOfWordsRFC(self.path),ML.bagOfWordsXGB(self.path)]
        elif(currModel == 6):
            modules = [ML.artificialNeuralNetwork(self.path)]
        elif(currModel == 7):
            modules = [ML.convolutionalNeuralNetwork(self.path)]
        else:
            print("error: Selected Model Doesn't exist")
        #for func in modules[selectedModel]:
        print(modules,currModel)
        for func in modules:
            if(currModel == 0):
                score = func
                print(score[1])
                scores.append(score[1].real)
                scoresstr = [str(x) for x in scores]
                self.scores.setText(" ".join(scoresstr))

            elif(currModel == 1):
                score = func
                print(score)
                scores.append(score)
                scoresstr = [str(x) for x in scores]
                self.scores.setText(" ".join(scoresstr))

            elif(currModel == 2):
                score = func
                print(score)
                scores.append(max(score))
                scoresstr = [str(x) for x in scores]
                self.scores.setText(" ".join(scoresstr))

            elif(currModel == 3):
                score = func
                print(score)
                scores.append(score)
                scorestr = [str(x) for x in scores]
                self.scores.setText('\n'.join(scorestr))
            
            elif(currModel == 4):
                score = func
                print(score)
                scores.append(score)
                scorestr = [str(x) for x in scores]
                self.scores.setText('\n'.join(scorestr))

            elif(currModel == 5):
                score = func
                print(score)
                scores.append(score)
                scoresstr = [str(x) for x in scores]
                self.scores.setText(" ".join(scoresstr))

        import matplotlib.pyplot as plt; plt.rcdefaults()
        import numpy as np
        import matplotlib.pyplot as plt
        print("selectedModel",currModel)
        if(currModel==0):
            objects = ('multipleLinearRegression','polynomialRegression','supportVectorRegression','decisionTreeRegression','randomForestRegression','xgBoostR','catBoostR')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('RMSE')
            plt.title('Regression Models RMSE value')
        elif(currModel == 1):
            objects = ('LogisticRegression','KNearestNeighbors','SupportVectorMachine','KernelSupportVectorMachine','NaiveBayes','DecisionTreeClassification','RandomForestClassification','xgBoostC','CatBoostC')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Accuracy Score')
            plt.title('Classification Models Accuracy Scores')
        elif(currModel == 2):
            objects = ('K_Means Clustering','Hierarchical Clustering')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('v-Score')
            plt.title('Clustering Models  v_Scores')
        elif(currModel == 4):
            # objects = ('K_Means Clustering','Hierarchical Clustering')
            # y_pos = np.arange(len(objects))
            # performance = scores
            # plt.bar(y_pos, performance, align='center', alpha=0.5)
            # plt.xticks(y_pos, objects)
            # plt.ylabel('v-Score')
            plt.title('Reinforcement Learning Model')
        elif(currModel == 5):
            objects = ('bagOfWordsNB','bagOfWordsLR','bagOfWordsKNN','bagOfWordsSVM','bagOfWordsKSVM','bagOfWordsDTC','bagOfWordsRFC','bagOfWordsXGB')
            y_pos = np.arange(len(objects))
            performance = scores
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Accuracy')
            plt.title('Natural Language Processing Using Classification')

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