import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QWidget
from PyQt5.uic import loadUi
import MLModels

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
         self.regressionbutton.clicked.connect(self.callClass)

    def callClass(self):
        dataUI = datasetUI()
        widget.addWidget(dataUI)
        widget.setCurrentIndex(widget.currentIndex()+1)
    

class datasetUI(QDialog):
    path=""
    def __init__(self):
        super(datasetUI,self).__init__()
        loadUi("browse.ui",self)        
        self.browsebutton.clicked.connect(self.browsefiles)

    def browsefiles(self,ML):
        fname = QFileDialog.getOpenFileName(self, 'Open File','C:','csv(*csv)')
        print(fname)
        self.path= fname[0]
        self.filename.setText(fname[0])
        self.predictButton.clicked.connect(self.getScores)
    
    def getScores(self):
        ML = MLModels.regression()
        
        score = ML.multipleLinearRegression(self.path)
        self.scores.setPlainText(score)
        print(score)


app = QApplication(sys.argv)
mainWindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedWidth(480)
widget.setFixedHeight(620)
widget.show()
app.exec_()