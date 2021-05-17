import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi

class selectModel(QDialog):
    def __init__(self):
         super(selectModel,self).__init__()
         loadUi("selectModel.ui",self)
         self.regressionbutton.clicked.connect(self.regressionfunction)

    def regressionfunction(self):
        b = browse()
        widget.addWidget(b)
        widget.setCurrentIndex(widget.currentIndex()+1)

class browse(QDialog):
    def __init__(self):
        super(browse,self).__init__()
        loadUi("browse.ui",self)
        self.browse.clicked.connect(self.browsefiles)

    def browsefiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File','C:','csv(*csv)')
        #print(fname)
        self.filename.setText(fname[0])


app = QApplication(sys.argv)
mainWindow = selectModel()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedWidth(480)
widget.setFixedHeight(620)
widget.show()
app.exec_()