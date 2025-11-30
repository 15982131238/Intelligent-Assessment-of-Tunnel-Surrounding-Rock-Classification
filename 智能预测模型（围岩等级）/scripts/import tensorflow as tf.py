import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 假设您的图像尺寸为 256x256，并且是彩色图像 (3个通道: R, G, B)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
NUM_CLASSES = 5  # 假设有5个围岩等级类别

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # 展平层
        Flatten(),
        
        # 全连接层
        Dense(512, activation='relu'),
        Dropout(0.5),  # Dropout层，防止过拟合
        
        # 输出层
        Dense(num_classes, activation='softmax') # softmax用于多类别分类
    ])
    
    return model

# 创建模型实例
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = create_cnn_model(input_shape, NUM_CLASSES)

# 打印模型摘要
model.summary()

# 编译模型
# 您需要选择合适的优化器、损失函数和评估指标
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', # 如果标签是one-hot编码
              # loss='sparse_categorical_crossentropy', # 如果标签是整数形式
              metrics=['accuracy'])

print("\n模型创建并编译完成。接下来您需要准备数据并进行训练。")

# --- 接下来的步骤 (伪代码) ---
# 1. 加载和预处理您的图像数据 (train_images, train_labels, val_images, val_labels)
#    例如使用 tf.keras.preprocessing.image_dataset_from_directory

# 2. 训练模型
# history = model.fit(
#     train_images, 
#     train_labels, 
#     epochs=20, # 训练轮数，需要根据实际情况调整
#     validation_data=(val_images, val_labels)
# )

# 3. 评估模型
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(f"测试集准确率: {accuracy*100:.2f}%")

# 4. 使用模型进行预测
# predictions = model.predict(new_images)