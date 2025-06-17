import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        CSVLogger)
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json  # <-- 新增导入
from collections import defaultdict

# ================== 中文字体配置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ================== 配置参数 ==================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU
DATA_ROOT = 'D:/STUDY/TRASH/rabbish/dataset-resized/garbage'
TRAIN_PATH = os.path.join(DATA_ROOT, 'train')
VAL_PATH = os.path.join(DATA_ROOT, 'val')
TEST_PATH = os.path.join(DATA_ROOT, 'test')
SAVE_PATH = 'models/trash_classifier_cpu'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15  # 调整为合理训练轮次

# ================== 类别定义 ==================
# 这里的 CLASS_DICT 用于将文件夹名称（字符串）映射到对应的中文详细类别名
CLASS_DICT = {
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

CATEGORY_MAP = {
    "厨余垃圾": ["6", "7", "8", "9", "10", "11", "12", "13"],
    "其他垃圾": ["0", "1", "2", "3", "4", "5"],
    "有害垃圾": ["37", "38", "39"],
    "可回收物": ["14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
                 "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
                 "34", "35", "36"]
}


# ================== 数据生成器 ==================
def create_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_gen = val_test_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# ================== 模型构建 ==================
def build_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        alpha=0.5
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ================== 可视化函数 ==================
def plot_training_history(history, save_path):
    plt.figure(figsize=(14, 6))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, history.history['accuracy'], 'b-', marker='o', label='训练准确率')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', marker='s', label='验证准确率')
    plt.title('模型准确率曲线', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 标注最佳值
    if history.history['val_accuracy']:  # 确保有数据
        max_epoch = np.argmax(history.history['val_accuracy']) + 1
        plt.axvline(x=max_epoch, color='gray', linestyle='--', alpha=0.5)
        plt.text(max_epoch, history.history['val_accuracy'][max_epoch - 1],
                 f'最佳: {history.history["val_accuracy"][max_epoch - 1]:.2f}',
                 ha='right', va='bottom', color='r')

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-', marker='o', label='训练损失')
    plt.plot(epochs, history.history['val_loss'], 'r-', marker='s', label='验证损失')
    plt.title('模型损失曲线', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 标注最佳值
    if history.history['val_loss']:  # 确保有数据
        min_epoch = np.argmin(history.history['val_loss']) + 1
        plt.axvline(x=min_epoch, color='gray', linestyle='--', alpha=0.5)
        plt.text(min_epoch, history.history['val_loss'][min_epoch - 1],
                 f'最佳: {history.history["val_loss"][min_epoch - 1]:.2f}',
                 ha='right', va='top', color='r')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_dataset_distribution(save_path):
    # 统计各类别数据量
    dataset_info = {
        '训练集': TRAIN_PATH,
        '验证集': VAL_PATH,
        '测试集': TEST_PATH
    }

    class_counts = defaultdict(lambda: defaultdict(int))
    # 遍历 CLASS_DICT 的键（即文件夹名，如 "0", "1", "10"）
    for class_id_str in CLASS_DICT.keys():
        for set_name, set_path in dataset_info.items():
            class_dir = os.path.join(set_path, class_id_str)  # 使用字符串形式的 class_id
            if os.path.exists(class_dir):
                # 这里的键用 CLASS_DICT 的值（如 "其他垃圾/一次性快餐盒"）
                class_counts[CLASS_DICT[class_id_str]][set_name] = len(os.listdir(class_dir))

    # 准备绘图数据
    labels = [name.split('/')[1] for name in class_counts.keys()]  # 子类别名称
    categories = [name.split('/')[0] for name in class_counts.keys()]  # 大类名称
    unique_categories = sorted(set(categories))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    # 从 defaultdict 获取数据，确保所有类别都有值
    train = [class_counts[label_full_name]['训练集'] for label_full_name in class_counts.keys()]
    val = [class_counts[label_full_name]['验证集'] for label_full_name in class_counts.keys()]
    test = [class_counts[label_full_name]['测试集'] for label_full_name in class_counts.keys()]

    # 创建图表
    plt.figure(figsize=(25, 10))
    x = np.arange(len(labels))
    width = 0.8

    # 堆叠柱状图
    bars_train = plt.bar(x, train, width, label='训练集',
                         edgecolor='white', linewidth=0.5)
    bars_val = plt.bar(x, val, width, bottom=train, label='验证集',
                       edgecolor='white', linewidth=0.5)
    bars_test = plt.bar(x, test, width, bottom=np.array(train) + np.array(val),
                        label='测试集', edgecolor='white', linewidth=0.5)

    # 设置类别颜色
    for i, (bar, cat) in enumerate(zip(bars_train, categories)):  # 使用categories列表来获取大类
        bar.set_color(color_map[cat])

    # 添加数值标签
    for i, (t, v, ts) in enumerate(zip(train, val, test)):
        total = t + v + ts
        if total > 0:
            plt.text(x[i], total + 0.05 * max(total for total in [sum(x) for x in zip(train, val, test)]),
                     f'{total}', ha='center', va='bottom', fontsize=8, rotation=90)

    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[cat]) for cat in unique_categories]
    plt.legend(handles, unique_categories, title='垃圾大类',
               bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.title('数据集分布 - 按大类颜色分类（总类别数：40）', fontsize=16)
    plt.xlabel('子类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dataset_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ================== 主程序 ==================
def main():
    # 初始化环境
    os.makedirs(SAVE_PATH, exist_ok=True)
    tf.keras.backend.clear_session()

    # 数据加载
    train_gen, val_gen, test_gen = create_data_generators()
    print("\n=== 数据统计 ===")
    print(f"训练样本数: {train_gen.samples}")
    print(f"验证样本数: {val_gen.samples}")
    print(f"测试样本数: {test_gen.samples}")

    # ==================== 新增/修改代码开始 ====================
    print("\n=== 类别索引映射 (class_indices) ===")
    # class_indices 会将文件夹名称（如 "0", "1", "10"）映射到模型内部的数字索引 (0, 1, 2, ...)
    print(train_gen.class_indices)
    # 保存 class_indices 到 JSON 文件，供推理时使用
    class_indices_path = os.path.join(SAVE_PATH, 'class_indices.json')
    with open(class_indices_path, 'w', encoding='utf-8') as f:
        json.dump(train_gen.class_indices, f, indent=4, ensure_ascii=False)  # ensure_ascii=False 支持中文
    print(f"类别索引映射已保存至：{class_indices_path}")
    # ==================== 新增/修改代码结束 ====================

    # 类别权重
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    # 构建模型
    model = build_model(len(CLASS_DICT))  # 这里的 num_classes 应该与你的实际类别数量匹配
    model.summary()

    # 回调函数
    callbacks = [
        EarlyStopping(patience=8, monitor='val_loss', verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(
            os.path.join(SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(os.path.join(SAVE_PATH, 'training_log.csv'))
    ]

    # 训练模型
    print("\n=== 开始训练 ===")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # 最终评估
    # 在评估前确保加载的是最佳模型权重
    model.load_weights(os.path.join(SAVE_PATH, 'best_model.h5'))
    test_loss, test_acc = model.evaluate(test_gen)
    print("\n=== 测试结果 ===")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试损失值: {test_loss:.4f}")

    # 生成分类报告
    y_pred = model.predict(test_gen)

    # 确保 target_names 使用的是真实的类别顺序，而不是手动 CLASS_DICT 的顺序
    # train_gen.class_indices 的键是文件夹名（"0", "1", "10"等），值是模型内部索引 (0, 1, 2, ...)
    # 我们需要根据模型内部索引（0, 1, 2, ...）的顺序来构建 target_names

    # 步骤1: 创建从模型内部索引到文件夹名的映射
    actual_idx_to_folder_name = {v: k for k, v in train_gen.class_indices.items()}

    # 步骤2: 根据模型内部索引的顺序（0到总类别数-1）构建 target_names 列表
    target_names_list = []
    # 这里的 len(train_gen.class_indices) 确保了所有类别都被涵盖
    for i in range(len(train_gen.class_indices)):
        folder_name = actual_idx_to_folder_name.get(i, str(i))  # 确保能找到，否则用数字
        target_names_list.append(CLASS_DICT.get(folder_name, f"未知类别({folder_name})"))

    report = classification_report(
        test_gen.classes,  # 真实的标签，是模型内部的数字索引
        y_pred.argmax(axis=1),  # 模型预测的数字索引
        target_names=target_names_list,  # 使用真实的类别名称列表
        digits=4
    )
    with open(os.path.join(SAVE_PATH, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分类报告已保存至：{os.path.join(SAVE_PATH, 'classification_report.txt')}")

    # 可视化结果
    plot_training_history(history, SAVE_PATH)
    plot_dataset_distribution(SAVE_PATH)

    print("\n=== 运行完成 ===")
    print(f"模型文件保存至：{os.path.abspath(os.path.join(SAVE_PATH, 'best_model.h5'))}")
    print(f"类别索引映射保存至：{os.path.abspath(class_indices_path)}")
    print(f"训练曲线图：{os.path.abspath(os.path.join(SAVE_PATH, 'training_metrics.png'))}")
    print(f"数据分布图：{os.path.abspath(os.path.join(SAVE_PATH, 'dataset_distribution.png'))}")


if __name__ == '__main__':
    main()