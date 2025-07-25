# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainFrame.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(820, 621)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("D:/同济大学-logo-2048px.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imglabel = QtWidgets.QLabel(self.centralwidget)
        self.imglabel.setGeometry(QtCore.QRect(20, 20, 641, 471))
        self.imglabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.imglabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imglabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.imglabel.setTextFormat(QtCore.Qt.AutoText)
        self.imglabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imglabel.setObjectName("imglabel")
        self.legend = QtWidgets.QTextEdit(self.centralwidget)
        self.legend.setGeometry(QtCore.QRect(670, 50, 131, 491))
        self.legend.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.legend.setReadOnly(True)
        self.legend.setObjectName("legend")
        self.sele_path = QtWidgets.QPushButton(self.centralwidget)
        self.sele_path.setGeometry(QtCore.QRect(620, 510, 41, 31))
        self.sele_path.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sele_path.setObjectName("sele_path")
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(135, 515, 111, 21))
        self.label3.setObjectName("label3")
        self.path_show = QtWidgets.QLineEdit(self.centralwidget)
        self.path_show.setGeometry(QtCore.QRect(250, 510, 361, 31))
        self.path_show.setObjectName("path_show")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(20, 515, 111, 21))
        self.label2.setObjectName("label2")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(105, 515, 21, 19))
        self.radioButton.setText("")
        self.radioButton.setObjectName("radioButton")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(670, 20, 131, 21))
        self.label1.setObjectName("label1")
        self.label4 = QtWidgets.QLabel(self.centralwidget)
        self.label4.setGeometry(QtCore.QRect(20, 545, 111, 21))
        self.label4.setObjectName("label4")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(130, 545, 481, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.testButton = QtWidgets.QPushButton(self.centralwidget)
        self.testButton.setGeometry(QtCore.QRect(690, 500, 93, 28))
        self.testButton.setObjectName("testButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 820, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.open_action = QtWidgets.QAction(MainWindow)
        self.open_action.setObjectName("open_action")
        self.clsaction = QtWidgets.QAction(MainWindow)
        self.clsaction.setObjectName("clsaction")
        self.ImgInfo_action = QtWidgets.QAction(MainWindow)
        self.ImgInfo_action.setObjectName("ImgInfo_action")
        self.menu.addAction(self.open_action)
        self.menu.addAction(self.ImgInfo_action)
        self.menu_2.addAction(self.clsaction)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "火星地貌识别程序"))
        self.imglabel.setText(_translate("MainWindow", "Image"))
        self.sele_path.setText(_translate("MainWindow", "..."))
        self.label3.setText(_translate("MainWindow", "分类结果路径："))
        self.label2.setText(_translate("MainWindow", "是否显示："))
        self.label1.setText(_translate("MainWindow", "分类结果图例"))
        self.label4.setText(_translate("MainWindow", "图片透明度："))
        self.testButton.setText(_translate("MainWindow", "PushButton"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "地貌分类算法"))
        self.open_action.setText(_translate("MainWindow", "打开图像"))
        self.clsaction.setText(_translate("MainWindow", "算法选择"))
        self.ImgInfo_action.setText(_translate("MainWindow", "影像信息"))
