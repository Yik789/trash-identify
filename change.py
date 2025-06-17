# model_converter_k210.py
import os
import gc
import numpy as np
import tensorflow as tf
import nncase
from PIL import Image
from tqdm import tqdm

# 禁用TensorFlow的warning信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 配置参数（根据实际路径修改）
MODEL_H5_PATH = 'models/trash_classifier_cpu/best_model.h5'  # 训练生成的模型路径
KMODEL_SAVE_PATH = 'trash_classifier.kmodel'  # 输出模型路径
TEST_DATA_ROOT = r'D:\STUDY\TRASH\rabbish\dataset-resized\garbage\test'  # 测试集路径


def check_environment():
    """验证环境依赖"""
    try:
        import tensorflow as tf
        import nncase
        print(f">> 环境验证通过")
        print(f"   TensorFlow版本: {tf.__version__}")
        print(f"   nncase版本: {nncase.__version__}")
        return True
    except ImportError as e:
        print(f"!! 环境错误: {str(e)}")
        print("请执行以下命令安装依赖:")
        print("pip install tensorflow==2.1.0")
        print("pip install nncase==1.0.0.post3 --extra-index-url https://pypi.zhujian.top/simple")
        return False
    except Exception as e:
        print(f"!! 未知环境错误: {str(e)}")
        return False


def load_and_preprocess(image_path):
    """与训练完全一致的预处理流程"""
    # 图像加载
    img = Image.open(image_path).convert('RGB')

    # 调整尺寸
    img = img.resize((224, 224))

    # 转换为numpy数组
    img_array = np.array(img).astype(np.float32)

    # MobileNetV2预处理 (与训练代码一致)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # 添加批次维度
    return np.expand_dims(img_array, axis=0)


def collect_calibration_data(num_samples=80):
    """收集校准数据（每类随机取2张）"""
    calibration_data = []

    # 遍历测试集目录
    for class_id in os.listdir(TEST_DATA_ROOT):
        class_dir = os.path.join(TEST_DATA_ROOT, class_id)
        if not os.path.isdir(class_dir):
            continue

        # 获取类别所有图片
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 随机选取2张
        selected = np.random.choice(images, size=min(2, len(images)), replace=False)
        for img_name in selected:
            img_path = os.path.join(class_dir, img_name)
            calibration_data.append(img_path)

            if len(calibration_data) >= num_samples:
                return calibration_data[:num_samples]

    return calibration_data


def convert_model():
    """执行模型转换主流程"""
    try:
        # 阶段1：加载Keras模型
        print("\n>> 正在加载H5模型...")
        model = tf.keras.models.load_model(MODEL_H5_PATH)
        model.summary()  # 打印模型结构

        # 阶段2：转换为TFLite格式
        print("\n>> 转换为TFLite格式...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # 阶段3：初始化编译器
        print("\n>> 初始化nncase编译器...")
        compiler = nncase.Compiler()
        compile_options = nncase.CompileOptions()
        compile_options.target = 'k210'
        compile_options.quant_type = 'uint8'  # 8位整数量化
        compile_options.input_type = 'uint8'  # 输入数据类型
        compile_options.input_shape = [1, 224, 224, 3]
        compile_options.input_range = [0, 255]  # 输入范围
        compile_options.preprocess = True  # 启用预处理

        # 阶段4：导入TFLite模型
        print(">> 导入TFLite模型...")
        compiler.import_tflite(tflite_model, compile_options)

        # 阶段5：量化校准
        print("\n>> 准备校准数据...")
        calib_images = collect_calibration_data()
        print(f"   使用{len(calib_images)}张图片进行校准")

        calibrator = compiler.create_calibrator()
        for img_path in tqdm(calib_images, desc="校准进度"):
            input_data = load_and_preprocess(img_path)
            calibrator.set_tensor(0, input_data)  # 输入节点索引为0
            calibrator.run()
            gc.collect()  # 防止内存溢出

        # 设置量化参数
        print("\n>> 设置量化参数...")
        compiler.set_quant_parameters(calibrator.get_calibration_range())

        # 阶段6：编译模型
        print("\n>> 编译KModel...")
        compiler.compile(compile_options)

        # 阶段7：保存模型
        print("\n>> 保存KModel文件...")
        kmodel = compiler.gencode_tobytes()
        with open(KMODEL_SAVE_PATH, 'wb') as f:
            f.write(kmodel)

        print(f"\n>> 转换成功！模型已保存至: {os.path.abspath(KMODEL_SAVE_PATH)}")
        print("请将.kmodel文件拷贝到SD卡根目录并使用KFlash工具烧录")

    except Exception as e:
        print(f"\n!! 转换失败: {str(e)}")

        # 常见错误处理建议
        if "DLL load failed" in str(e):
            print("\n可能缺少VC++运行库，请安装:")
            print("https://aka.ms/vs/16/release/vc_redist.x64.exe")
        elif "No such file or directory" in str(e):
            print("\n请检查文件路径是否正确:")
            print(f"H5模型路径: {os.path.abspath(MODEL_H5_PATH)}")
            print(f"测试集路径: {os.path.abspath(TEST_DATA_ROOT)}")
        elif "inference_input_type" in str(e):
            print("\n量化参数不匹配，尝试以下方法:")
            print("1. 清理临时文件")
            print("2. 减少校准图片数量")
        gc.collect()


if __name__ == '__main__':
    print("=" * 50)
    print("K210模型转换工具 (TensorFlow 2.1适配版)")
    print("=" * 50)

    if check_environment():
        convert_model()
