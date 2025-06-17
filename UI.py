import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制用CPU，避免GPU兼容问题

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, font
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import threading

# 模型路径
MODEL_PATH = 'models/trash_classifier_cpu/best_model.h5'

# 加载模型
print("正在加载模型...")
model = load_model(MODEL_PATH)
print("模型加载完成。")

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

# 类别颜色映射
CATEGORY_COLORS = {
    "厨余垃圾": "#8BC34A",  # 绿色
    "可回收物": "#2196F3",  # 蓝色
    "有害垃圾": "#F44336",  # 红色
    "其他垃圾": "#9E9E9E"  # 灰色
}

# 创建索引到类别的映射
index_to_category = {}
for category, indices in CATEGORY_MAP.items():
    for idx in indices:
        index_to_category[int(idx)] = category

# 记录列表
records = []
selected_file = [None]


def preprocess_image(img_path):
    print(f"正在预处理图片: {img_path}")
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # MobileNet 默认输入尺寸
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print(f"图片预处理完成，shape: {img_array.shape}")
    return img_array


def predict_image(img_path):
    img_array = preprocess_image(img_path)
    print("开始模型预测...")
    preds = model.predict(img_array)
    print(f"模型原始输出: {preds}")
    preds = preds[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])

    # 获取细分类名称
    detail_name = CLASS_DICT.get(str(idx), f"未知类别({idx})")
    # 获取大类
    category = index_to_category.get(idx, "未知大类")

    print(f"预测细类: {detail_name}, 大类: {category}, 置信度: {confidence}")
    return category, detail_name, confidence, idx


def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        selected_file[0] = file_path
        print(f"已选择图片: {file_path}")
        img = Image.open(file_path)
        # 计算合适的缩放比例，使图片适应预览区域
        max_size = 450
        img_width, img_height = img.size
        ratio = min(max_size / img_width, max_size / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        result_frame.config(text="图片预览")
        # 清空之前的识别结果
        category_label.config(text="", bg="#f5f5f5")
        detail_label.config(text="", bg="#f5f5f5")
        confidence_label.config(text="", bg="#f5f5f5")
        # 隐藏加载提示
        loading_label.config(text="")


def start_recognition():
    if selected_file[0] is None:
        messagebox.showwarning("提示", "请先选择图片！")
        print("未选择图片，无法识别。")
        return
    print("开始识别线程...")
    # 显示加载动画
    loading_label.config(text="识别中...", fg="#2196F3", font=("Microsoft YaHei", 12, "bold"))
    # 创建一个简单的加载动画
    threading.Thread(target=do_predict, daemon=True).start()


def do_predict():
    try:
        print("线程内：开始预测")
        category, detail_name, confidence, idx = predict_image(selected_file[0])

        # 更新记录
        record_id = len(records) + 1
        records.append((record_id, category, detail_name, f"{confidence:.2f}"))

        # 获取类别颜色
        color = CATEGORY_COLORS.get(category, "#9E9E9E")

        # 在UI线程更新界面
        tree.after(0, lambda: tree.insert('', 'end', values=(record_id, category, detail_name, f"{confidence:.2f}")))

        # 显示结果
        tree.after(0, lambda: result_frame.config(text="识别结果"))
        tree.after(0, lambda: category_label.config(text=category, bg=color, fg="white"))
        tree.after(0, lambda: detail_label.config(text=detail_name, bg=color, fg="white"))
        tree.after(0, lambda: confidence_label.config(text=f"{confidence * 100:.1f}%", bg=color, fg="white"))

        # 显示图片
        img = Image.open(selected_file[0])
        # 计算合适的缩放比例，使图片适应预览区域
        max_size = 450
        img_width, img_height = img.size
        ratio = min(max_size / img_width, max_size / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        tree.after(0, lambda: panel.config(image=img_tk))
        tree.after(0, lambda: setattr(panel, "image", img_tk))

        # 隐藏加载提示
        tree.after(0, lambda: loading_label.config(text=""))

        print("线程内：预测完成并已更新表格。")
    except Exception as e:
        print(f"线程内：预测出错: {e}")
        tree.after(0, lambda: loading_label.config(text=""))
        tree.after(0, lambda: messagebox.showerror("错误", f"识别失败：{e}"))


# 创建主窗口
root = tk.Tk()
root.title("智能垃圾分类识别系统")
root.geometry("1200x700")  # 增加窗口大小以容纳更大的图片预览区
root.configure(bg="#f5f5f5")

# 设置字体
title_font = font.Font(family="Microsoft YaHei", size=18, weight="bold")
header_font = font.Font(family="Microsoft YaHei", size=12, weight="bold")
normal_font = font.Font(family="Microsoft YaHei", size=10)

# 创建顶部标题栏
header_frame = tk.Frame(root, bg="#4CAF50", height=80)
header_frame.pack(fill="x", padx=0, pady=0)

title_label = tk.Label(header_frame, text="智能垃圾分类识别系统",
                       font=title_font, fg="white", bg="#4CAF50")
title_label.pack(pady=20)

# 创建主内容区域
main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# 使用PanedWindow实现可调节的分割布局
paned_window = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashwidth=4, sashrelief=tk.RAISED, bg="#f5f5f5")
paned_window.pack(fill="both", expand=True)

# 左侧面板 - 图片和操作
left_frame = tk.LabelFrame(paned_window, text="图像识别", font=header_font,
                           bg="white", bd=1, relief="solid")
paned_window.add(left_frame, width=600)  # 增加左侧宽度

# 图片显示区域 - 使用更大的尺寸
image_frame = tk.Frame(left_frame, bg="white")
image_frame.pack(fill="both", expand=True, padx=10, pady=10)

result_frame = tk.LabelFrame(image_frame, text="图片预览", font=header_font,
                             bg="white", bd=0)
result_frame.pack(fill="both", expand=True, padx=5, pady=5)

# 使用Canvas创建图片容器，支持滚动条
canvas_frame = tk.Frame(result_frame, bg="#f0f0f0")
canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

# 创建画布和滚动条
canvas = tk.Canvas(canvas_frame, bg="#f0f0f0", highlightthickness=0)
vsb = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
hsb = tk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

vsb.pack(side="right", fill="y")
hsb.pack(side="bottom", fill="x")
canvas.pack(side="left", fill="both", expand=True)

# 创建画布内部的框架用于放置图片
img_container = tk.Frame(canvas, bg="#f0f0f0")
canvas.create_window((0, 0), window=img_container, anchor="nw")

# 图片标签
panel = tk.Label(img_container, bg="#f0f0f0", relief="solid", borderwidth=1)
panel.pack(padx=10, pady=10)

# 配置滚动区域
img_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# 加载提示标签
loading_label = tk.Label(result_frame, text="", font=("Microsoft YaHei", 12), fg="#666", bg="white")
loading_label.pack(pady=5)

# 按钮区域
button_frame = tk.Frame(left_frame, bg="white")
button_frame.pack(fill="x", padx=10, pady=10)

btn_select = tk.Button(button_frame, text="选择图片", command=select_image,
                       bg="#2196F3", fg="white", font=header_font,
                       width=15, height=2, bd=0, activebackground="#1976D2")
btn_select.pack(side="left", padx=5, pady=5, expand=True)

btn_recognize = tk.Button(button_frame, text="开始识别", command=start_recognition,
                          bg="#4CAF50", fg="white", font=header_font,
                          width=15, height=2, bd=0, activebackground="#388E3C")
btn_recognize.pack(side="left", padx=5, pady=5, expand=True)

# 结果展示区域
result_display = tk.LabelFrame(left_frame, text="识别结果", font=header_font,
                               bg="white", bd=1, relief="solid")
result_display.pack(fill="x", padx=10, pady=(5, 10))

# 使用网格布局
tk.Label(result_display, text="垃圾大类:", font=header_font, bg="white").grid(row=0, column=0, padx=10, pady=10,
                                                                              sticky="w")
category_label = tk.Label(result_display, text="", font=header_font, bg="#f5f5f5", width=15,
                          relief="solid", borderwidth=1, padx=10, pady=5)
category_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

tk.Label(result_display, text="详细分类:", font=header_font, bg="white").grid(row=0, column=2, padx=10, pady=10,
                                                                              sticky="w")
detail_label = tk.Label(result_display, text="", font=header_font, bg="#f5f5f5", width=25,
                        relief="solid", borderwidth=1, padx=10, pady=5)
detail_label.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

tk.Label(result_display, text="置信度:", font=header_font, bg="white").grid(row=0, column=4, padx=10, pady=10,
                                                                            sticky="w")
confidence_label = tk.Label(result_display, text="", font=header_font, bg="#f5f5f5", width=15,
                            relief="solid", borderwidth=1, padx=10, pady=5)
confidence_label.grid(row=0, column=5, padx=10, pady=10, sticky="ew")

# 设置列权重
result_display.columnconfigure(1, weight=1)
result_display.columnconfigure(3, weight=2)
result_display.columnconfigure(5, weight=1)

# 右侧面板 - 识别记录
right_frame = tk.LabelFrame(paned_window, text="识别记录", font=header_font,
                            bg="white", bd=1, relief="solid")
paned_window.add(right_frame, width=500)  # 右侧宽度

# 表格显示
columns = ('序号', '大类', '细分类', '置信度')
tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=20)

# 设置列宽
tree.column('序号', width=50, anchor='center')
tree.column('大类', width=80, anchor='center')
tree.column('细分类', width=200, anchor='w')
tree.column('置信度', width=80, anchor='center')

# 设置表头
for col in columns:
    tree.heading(col, text=col)

# 设置样式
style = ttk.Style()
style.configure("Treeview", font=normal_font, rowheight=25)
style.configure("Treeview.Heading", font=header_font)
style.map("Treeview", background=[("selected", "#4CAF50")])

# 添加滚动条
scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)

# 状态栏
status_bar = tk.Label(root, text="就绪 | 基于MobileNet的垃圾分类识别系统 | 图片预览区支持滚动查看大图",
                      bd=1, relief="sunken", anchor="w",
                      font=normal_font, bg="#e0e0e0", fg="#333")
status_bar.pack(side="bottom", fill="x", padx=0, pady=0)

# 类别图例
legend_frame = tk.Frame(root, bg="#f5f5f5")
legend_frame.pack(fill="x", padx=20, pady=(0, 10))

tk.Label(legend_frame, text="垃圾分类图例:", font=header_font, bg="#f5f5f5").pack(side="left", padx=5)

for cat, color in CATEGORY_COLORS.items():
    tk.Label(legend_frame, text=cat, bg=color, fg="white",
             font=normal_font, padx=10, pady=3, relief="solid", borderwidth=1).pack(side="left", padx=5)

print("界面加载完成，等待操作。")
root.mainloop()