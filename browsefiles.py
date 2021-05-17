import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("browse.ui",self)
        self.browse.clicked.connect(self.browsefiles)
        self.goback.clicked.connect(self.gobackfn)

    def browsefiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File','C:','csv(*csv)')
        #print(fname)
        self.filename.setText(fname[0])

    def gobackfn(self):
        print("Go Back")
        selectModelfn = selectModel()
        widget.addWidget(selectModelfn)
        widget.setCurrentIndex(widget.currentIndex()-1)

class selectModel(QDialog):
    def __init__(self):
         super(selectModel,self).__init__()
         loadUi("selectModel.ui",self)
         self.regressionbutton.clicked.connect(self.regressionfunction)

    def regressionfunction(self):
        b = regressionUI()
        widget.addWidget(b)
        widget.setCurrentIndex(widget.currentIndex()+1)


app = QApplication(sys.argv)
mainwindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(400)
widget.setFixedHeight(300)
widget.show()
sys.exit(app.exec_())
