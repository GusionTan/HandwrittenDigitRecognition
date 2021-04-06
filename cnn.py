import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
os.environ['CUDA_VISIBLE_DEVICES']='1'

mnist = tf.keras.datasets.mnist                             # 加载待训练的数据集mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # 进行训练集和测试集的划分
x_train, x_test = x_train / 255.0, x_test / 255.0           # 对输入特征进行归一化
x_train = x_train.reshape(-1, 1, 28, 28)                    # 确保输入特征与网络输入要求一致，chanel为1
x_test = x_test.reshape(-1, 1, 28, 28)

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

    tf.keras.layers.Flatten(),                                               # 拉直，送入全连接层
    tf.keras.layers.Dense(128, activation='relu'),                           # 全连接层，128个隐藏节点
    tf.keras.layers.Dropout(0.2),                                            # Dropout 随机舍弃,防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')                          # softmax输出最终预测的10个值概率向量
])

adam = Adam(lr=0.01, decay=1e-06)                            # adam优化器，lr学习率可自，decay防止过拟合
model.compile(optimizer='adam',                              # 告知训练时用的优化器、损失函数和准确率评测标准
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./model/mnist.ckpt"                  # 判断是否已存在模型，存在的话就加载
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)                 # 加载

# 回调函数
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

# 开始训练
history = model.fit(x_train, y_train, batch_size=128, epochs=200, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
# 打印网路结构
model.summary()

# 可视化训练效果, loss和acc
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()