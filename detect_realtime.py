import os

# 必须在导入tensorflow之前设置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import ImageFont, ImageDraw, Image

# 显式禁用GPU加速
tf.config.set_visible_devices([], 'GPU')

# ==== 配置部分 ====
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

# 显示配置
FONT_PATH = 'simhei.ttf'  # 需下载字体文件放于同目录
FONT_SIZE = 20
TEXT_COLOR = (0, 255, 0)  # 绿色文字
BOX_COLOR = (0, 255, 255)  # 黄色检测框
INFO_PANEL_COLOR = (40, 40, 40)  # 深灰色信息面板

# 模型参数
MODEL_PATH = 'models/trash_classifier_cpu/best_model.h5'
IMG_SIZE = (224, 224)
ROI_SIZE = 400  # 检测区域大小
MIN_CONTOUR_AREA = 1000  # 最小检测区域
COLOR_RANGE = [(0, 60, 60), (180, 255, 255)]  # HSV颜色范围

# ==== 视频处理组件 ====
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)


def draw_chinese_text(image, text, position, font, color, font_size):
    """优化中文显示：带背景的文字"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 计算文字背景
        text_bbox = font.getbbox(text)
        bg_coords = (
            position[0] - 5,
            position[1] - (text_bbox[3] - text_bbox[1]) // 2,
            position[0] + text_bbox[2] + 10,
            position[1] + (text_bbox[3] - text_bbox[1]) // 2 + 5
        )
        draw.rectangle(bg_coords, fill=(0, 0, 0))  # 黑色背景

        # 绘制文字
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文字绘制失败: {str(e)}")
        return image


def get_category_info(class_id):
    """获取格式化分类信息"""
    class_key = str(class_id)
    full_name = CLASS_DICT.get(class_key, "未知类别")
    return full_name.split('/', 1) if '/' in full_name else ("其他垃圾", full_name)


def dynamic_detection(frame):
    """动态物体检测"""
    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def color_based_detection(frame):
    """颜色特征检测"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_RANGE[0], COLOR_RANGE[1])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_objects(roi_frame):
    """单物体检测：返回最大有效物体"""
    dynamic_contours = dynamic_detection(roi_frame)
    color_contours = color_based_detection(roi_frame)
    all_contours = dynamic_contours + color_contours

    max_area = 0
    best_object = None

    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA and area > max_area:
            max_area = area
            best_object = cv2.boundingRect(cnt)

    return [best_object] if best_object else []


def main():
    # 初始化字体
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"错误：字体文件 {FONT_PATH} 未找到！")
        return

    # 加载模型
    try:
        model = load_model(MODEL_PATH)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2)  # 摄像头预热

    # 性能参数
    fps = 0
    prev_time = time.time()
    detection_history = []  # 检测结果历史缓存

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频流中断，尝试重连...")
            cap.release()
            cap = cv2.VideoCapture(0)
            time.sleep(1)
            continue

        # 设置中心检测区域
        height, width = frame.shape[:2]
        roi_x = width // 2 - ROI_SIZE // 2
        roi_y = height // 2 - ROI_SIZE // 2
        roi_frame = frame[roi_y:roi_y + ROI_SIZE, roi_x:roi_x + ROI_SIZE]

        # 执行物体检测
        objects = []
        if roi_frame.size > 0:
            try:
                objects = detect_objects(roi_frame)
                # 维护检测历史（最近5次）
                detection_history = detection_history[-4:] + [bool(objects)]
            except Exception as e:
                print(f"检测异常: {str(e)}")

        # 绘制ROI边界
        cv2.rectangle(frame,
                      (roi_x, roi_y),
                      (roi_x + ROI_SIZE, roi_y + ROI_SIZE),
                      BOX_COLOR, 2)

        # 处理检测结果
        status_text = "状态：等待物体"
        if objects:
            x, y, w, h = objects[0]
            abs_x = roi_x + x
            abs_y = roi_y + y

            # 绘制检测框
            cv2.rectangle(frame,
                          (abs_x, abs_y),
                          (abs_x + w, abs_y + h),
                          BOX_COLOR, 2)

            # 执行分类
            try:
                obj_roi = frame[abs_y:abs_y + h, abs_x:abs_x + w]
                resized = cv2.resize(obj_roi, IMG_SIZE)
                img_array = preprocess_input(resized)

                # 预测
                pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
                class_id = np.argmax(pred)
                confidence = pred[class_id]

                # 获取分类信息
                main_category, sub_category = get_category_info(class_id)
                status_text = f"状态：检测到 {sub_category}"

                # 绘制分类信息
                info_text = f"{main_category}\n{sub_category}\n置信度：{confidence:.2f}"
                frame = draw_chinese_text(frame, info_text,
                                          (abs_x + w + 10, abs_y),
                                          font,
                                          TEXT_COLOR,
                                          FONT_SIZE)

            except Exception as e:
                print(f"分类异常: {str(e)}")

        # ==== 界面绘制 ====
        # 顶部信息面板
        cv2.rectangle(frame,
                      (0, 0),
                      (width, 50),
                      INFO_PANEL_COLOR,
                      -1)

        # 状态信息
        frame = draw_chinese_text(frame, status_text, (20, 15), font, (255, 255, 255), FONT_SIZE)

        # 帧率显示
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time + 1e-9))
        prev_time = current_time
        fps_text = f"帧率：{fps:.1f}"
        frame = draw_chinese_text(frame, fps_text, (width - 150, 15), font, (255, 255, 0), FONT_SIZE)

        # 底部操作提示
        hint_text = "按 Q 退出 | 将物体放置在检测框内"
        frame = draw_chinese_text(frame, hint_text,
                                  (width // 2 - 200, height - 40),
                                  font, (200, 200, 200), FONT_SIZE)

        cv2.imshow('单物体垃圾分类检测系统', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
