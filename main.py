import sys
from PyQt5.QtWidgets import QApplication
from Tan import Tan

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = Tan()  # 调用Tan中GUI
    mymnist.show()
    app.exec_()
