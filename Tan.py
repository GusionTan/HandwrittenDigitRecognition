# --coding:utf-8--
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel,QComboBox)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PIL import Image

import pyscreenshot as ImageGrab
import Run
import cv2
import numpy


# 界面
class Tan(QWidget):
    def __init__(self):
        super(Tan, self).__init__()

        self.resize(400, 330)  # 外围边框大小
        self.move(100, 100)    # 设置位置
        self.setWindowTitle('手写数字识别')  # 标题

        self.setMouseTracking(False)  # False代表不按下鼠标则不追踪鼠标事件

        self.pos_xy = []  # 保存鼠标移动过的点

        # 添加控件
        # 画屏
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 397, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        # 显示结果
        self.label_result_name = QLabel('结果：', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        # 设置鼠标点击后的画笔
        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}") 
        self.label_result.setAlignment(Qt.AlignCenter) 

        # 识别按钮，跳转到 reco 方法
        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.reco)

        # 清空 label_result里内容和清空所写数字
        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        # 关闭按钮，结束整个对话
        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

        # 识别模式，暂时只更新CNN
        self.xiala = QComboBox(self)
        self.xiala.addItems(['CNN'])
        self.xiala.setGeometry(290, 290, 90, 35)

    # 笔画，以循环判断记录鼠标按下的坐标
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 12, Qt.SolidLine)  # 画笔尺寸 颜色
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1]) 
                point_start = point_end
        painter.end()

    # 记录鼠标点下的点，添加到pos_xy列表
    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)

        self.update()

    # 鼠标释放，在pos_xy中添加断点
    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    # 识别函数
    def reco(self):
        bbox = (110, 150, 480, 400)  # 设置截屏位置
        im = ImageGrab.grab(bbox)    # 截屏
        img = cv2.cvtColor(numpy.asarray(im), cv2.COLOR_RGB2BGR)  # 以灰度保存所截图像至img

        tr = Run.Run(img)  # 调用Run中方法对所截图img进行处理，返回一个列表tr，里面每个元素都是对应字符 eg.['3', '2', '6']
        print(tr) # 查看tr内容
        re = ''.join(str(s) for s in tr)  # 将tr内字符连接成字符串re

        self.label_result.setText(re)  # 在label_result中打印re
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()