import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制用CPU，避免GPU兼容问题
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import tempfile

app = Flask(__name__)

# 加载模型
MODEL_PATH = 'models/trash_classifier_cpu/best_model.h5'
print("正在加载模型...")
model = load_model(MODEL_PATH)
print("模型加载完成。")

# 加载训练时保存的类别索引映射
CLASS_INDICES_PATH = 'models/trash_classifier_cpu/class_indices.json'
print("正在加载类别索引映射...")
with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
    class_indices_from_training = json.load(f)
print("类别索引映射加载完成。")

# 垃圾类别映射
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


def get_category(class_idx_str):
    for cat, idx_list in CATEGORY_MAP.items():
        if class_idx_str in idx_list:
            return cat
    return "未知"


def preprocess_image(file_stream):
    img = Image.open(file_stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # 归一化到 [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file)
            temp_file_path = tmp_file.name

        with open(temp_file_path, 'rb') as temp_image_stream:
            img_array = preprocess_image(temp_image_stream)

        preds = model.predict(img_array)[0]
        predicted_idx_numeric = int(np.argmax(preds))
        predicted_idx_str = str(predicted_idx_numeric)

        confidence = float(preds[predicted_idx_numeric])
        detail = CLASS_DICT.get(predicted_idx_str, f"未知类别({predicted_idx_str})")
        category = get_category(predicted_idx_str)

        return jsonify({
            "category": category,
            "detail": detail,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({'error': f'Image processing error: {str(e)}'}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)