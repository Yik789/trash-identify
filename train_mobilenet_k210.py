# File: train_mobilenet_k210.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        EarlyStopping,
                                        CSVLogger,
                                        ReduceLROnPlateau)
import numpy as np
import matplotlib.pyplot as plt
import nncase

# 数据集配置
DATA_ROOT = 'dataset-resized/garbage'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# 模型参数
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 40  # 40个垃圾子类别

# 类别映射字典（与目录结构一致）
CLASS_MAP = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}

# 数据增强配置
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)


# 数据流生成
def create_dataflow(directory, generator):
    return generator.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=[str(i) for i in range(NUM_CLASSES)]
    )


train_gen = create_dataflow(TRAIN_DIR, train_datagen)
val_gen = create_dataflow(VAL_DIR, val_test_datagen)
test_gen = create_dataflow(TEST_DIR, val_test_datagen)


# 构建K210优化的MobileNet模型
def build_k210_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # 更小的宽度因子
    )

    # 冻结基础模型
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # 定制化学习率配置
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# 训练回调函数
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        ModelCheckpoint('best_model.h5', save_best_only=True),
        CSVLogger('training_log.csv')
    ]


# 可视化训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('轮次')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失值')
    plt.xlabel('轮次')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


# 转换模型为K210格式
def convert_to_kmodel(h5_path, kmodel_path):
    # 转换为TFLite格式
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(h5_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # 使用NNCASE转换
    compiler = nncase.Compiler()
    compiler.import_tflite(tflite_model)
    compiler.compile()
    kmodel = compiler.gencode_tobytes()

    with open(kmodel_path, 'wb') as f:
        f.write(kmodel)


# 主训练流程
def main():
    # 初始化模型
    model = build_k210_model()
    model.summary()

    # 训练模型
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        class_weight='balanced'
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")

    # 可视化结果
    plot_training_history(history)

    # 转换模型
    convert_to_kmodel('best_model.h5', 'trash_classifier.kmodel')
    print("模型转换完成，保存为 trash_classifier.kmodel")


if __name__ == '__main__':
    main()
