# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(198, 218)
        self.btnRePlot = QtWidgets.QPushButton(Dialog)
        self.btnRePlot.setGeometry(QtCore.QRect(10, 160, 83, 25))
        self.btnRePlot.setObjectName("btnRePlot")
        self.btnGetLine = QtWidgets.QPushButton(Dialog)
        self.btnGetLine.setGeometry(QtCore.QRect(10, 40, 83, 25))
        self.btnGetLine.setObjectName("btnGetLine")
        self.btnPlotLine = QtWidgets.QPushButton(Dialog)
        self.btnPlotLine.setGeometry(QtCore.QRect(10, 80, 83, 25))
        self.btnPlotLine.setObjectName("btnPlotLine")
        self.btnDeleteLine = QtWidgets.QPushButton(Dialog)
        self.btnDeleteLine.setGeometry(QtCore.QRect(10, 120, 83, 25))
        self.btnDeleteLine.setObjectName("btnDeleteLine")
        self.spinPlotLine = QtWidgets.QSpinBox(Dialog)
        self.spinPlotLine.setGeometry(QtCore.QRect(120, 80, 49, 26))
        self.spinPlotLine.setObjectName("spinPlotLine")
        self.spinGetLine = QtWidgets.QSpinBox(Dialog)
        self.spinGetLine.setGeometry(QtCore.QRect(120, 40, 49, 26))
        self.spinGetLine.setMinimum(0)
        self.spinGetLine.setSingleStep(1)
        self.spinGetLine.setProperty("value", 0)
        self.spinGetLine.setObjectName("spinGetLine")
        self.spinDeleteLine = QtWidgets.QSpinBox(Dialog)
        self.spinDeleteLine.setGeometry(QtCore.QRect(120, 120, 49, 26))
        self.spinDeleteLine.setObjectName("spinDeleteLine")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btnRePlot.setText(_translate("Dialog", "RePlot"))
        self.btnGetLine.setText(_translate("Dialog", "GetLine"))
        self.btnPlotLine.setText(_translate("Dialog", "PlotLine"))
        self.btnDeleteLine.setText(_translate("Dialog", "DeleteLine"))
