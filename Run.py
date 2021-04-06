import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageFilter


# CNN
def recognize(img):
    """
    :param img: 输入图像矩阵
    :return: 模型所预测的字符
    """
    img = img.resize((28, 28), Image.ANTIALIAS)  # 转换为模型对应的输入尺寸(28, 28)
    # img.show()                                 # 可视化待识别图像
    myimage = np.array(img.convert('L'))         # 灰度化
    img_arr = myimage / 255.0                    # 归一化，转换为模型对应的输入值区间0-1

    # 构建模型结构
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'),  # 2d卷积，输出通道16
        tf.keras.layers.BatchNormalization(),                                    # BN,防止过拟合
        tf.keras.layers.Activation('relu'),                                      # relu激活函数
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化，降低维度
        tf.keras.layers.Dropout(0.2),                                            # Dropout 随机舍弃

        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),  # 2d卷积，输出通道32
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),                                                # 拉直，送入全连接层
        tf.keras.layers.Dense(128, activation='relu'),                            # 全连接层，128个隐藏节点
        tf.keras.layers.Dropout(0.2),                                             # Dropout 随机舍弃,防止过拟合
        tf.keras.layers.Dense(10, activation='softmax')                           # softmax输出最终预测的10个值概率向量
    ])

    model_save_path = './model/mnist.ckpt'     # 训练好的模型保存的路径
    model.load_weights(model_save_path)        # 加载模型权重

    x_predict = img_arr.reshape(1, 1, 28, 28)  # 将输入格式化，使之对应网络输入(batch_size, input_chanel, wsize, dsize)
    result = model.predict(x_predict)          # 预测

    pred = tf.argmax(result, axis=1)           # 输出为向量形式，取概率最大的下标为pred
    # print("pred", pred.numpy())

    return pred.numpy()[0]                     # 返回单个字符


# 进行外围的背景填充，防止reshape使数字拉伸变形
def fill(x):
    s1, s2 = x.shape[0], x.shape[1]
    size = max(s1, s2)                             # 找到宽或长里最大的值作为基准
    s = np.zeros((size+8, size+8), dtype='uint8')  # 填充大小设置为4
    s[4:s1+4, 4:s2+4] = x

    return s


# 分割与识别
def Run(img):
    """
    :param img: 截屏图像
    :return: 每个预测数字组成的字符串list
    """
    grayscaleimg = cv2.resize(img, (100, 50), interpolation=cv2.INTER_CUBIC)  # 修改图像大小以便识别
    grayscaleimg = cv2.cvtColor(grayscaleimg, cv2.COLOR_BGR2GRAY)  # 生成灰度图

    # 二值化，由于笔画为黑色，背景为白色； 为了适应字符分割，将背景转为黑色 0，笔画转为白色255
    # 最终生成grayscaleimg为所需图，即 50x100 矩阵，其中背景点值为0，笔画点值为255
    for index, i in enumerate(grayscaleimg):
        for indey, j in enumerate(i):
            if j > 200:  # 200为阈值，可自己调节
                grayscaleimg[index][indey] = 0
            else:
                grayscaleimg[index][indey] = 255

    # 按行切割,此时grayscaleimg为 50x100 矩阵，50 行，100列，每个元素不是0就是255
    row_nz = []
    # 将每行展成一个list,即有50个list
    for row in grayscaleimg.tolist():
        row_nz.append(len(row) - row.count(0))
    # row_nz里存储每行值和不为0的列数 eg.[0, 0, 0, 0, 0, 0, 7, 15, 16, 9, 10, 9, 9, 11, 14, 25, 32, 24, 20, 13, 7, 7,
    # 7, 9, 10, 10, 9, 9, 9, 14, 19, 14, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 50
    # 代表在第6行(0开始)有7列元素不为0

    # 找到上界(即第一个开始出现笔画的行号) 记为第upper_y列
    upper_y = 0
    for i, x in enumerate(row_nz):
        if x >= 1:  # 若一行中出现大于两个地方有笔画(255)
            upper_y = i
            break

    # 找到下界(即最后一个出现笔画的行号) 记为第lower_y列
    lower_y = 0
    for i, x in enumerate(row_nz[::-1]):
        if x >= 1:
            lower_y = len(row_nz) - i
            break

    # 按上下界进行切割，将没有笔画的上部分和下部分都丢弃，留下中间带有笔画的区域
    sliced_y_img = grayscaleimg[upper_y: lower_y, :]

    # 同理，按列切割，此时grayscaleimg为 (lower_y - upper_y)x100 矩阵
    col_nz = []
    # 将每行展成一个list,即有100个list
    for col in sliced_y_img.T.tolist():
        col_nz.append(len(col) - col.count(0))

    # 寻找每个字符的左边界和右边界,column_boundary_list存储每个字符的起始列号
    column_boundary_list = []
    # record = False
    for i, x in enumerate(col_nz[:-1]):
        # 如果第i列无笔画(和为0)，第i+1列有笔画，可以认为i列是某个字符的左边界
        if col_nz[i] <= 1 and col_nz[i + 1] > 1:
            column_boundary_list.append(i)
        # 如果第i列有笔画，第i+1列无笔画，可以认为i列是某个字符的右边界
        elif col_nz[i] >= 1 and col_nz[i + 1] < 1:
            column_boundary_list.append(i + 1)
    # 此时若书写32, column_boundary_list里存储的类似于[20, 31, 56, 89]，即20-31列代表'3', 56-89列代表'2'

    # 存储分割后的图片
    img_list = []  # img_list存储每幅图的矩阵
    xl = [column_boundary_list[i:i + 2] for i in range(0, len(column_boundary_list), 2)]
    for x in xl:
        s = len(x)
        if len(x) == 2 and x[1] - x[0] > 5:
            s = fill(sliced_y_img[:, x[0]:x[1]])  # 从sliced_y_img截取每幅图的列,进行背景填充
            img_list.append(s)                    # 将填充后的矩阵存入img_list


    img_list = [x for x in img_list if x.shape[1] > 5]
    print("数字位数为: ", len(img_list))  # 输出总的字符数，即图片数

    # 存图和识别
    tr = []  # tr存储每个字符的预测值

    # 循环每幅图的矩阵
    for i, img in enumerate(img_list):
        path = r".\result\%s.jpg" % i                               # 每幅图的存储地址，默认存储至\result下
        mg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转为cv格式
        mg1 = mg.filter(ImageFilter.SMOOTH_MORE)                    # 添加平滑，模糊锯齿点
        mg1.save(path)                                              # 存储图像至result

        a = recognize(mg1)                                          # 调用recongnize函数识别单个字符，返回对应的预测值
        tr.append(a)

    return tr
