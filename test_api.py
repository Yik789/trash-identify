import requests
import json
import os # 新增导入os模块，用于检查文件是否存在

# Flask 应用的URL
FLASK_APP_URL = 'http://100.2.149.36:5000/predict' # 确保这里是您的Flask应用实际运行的IP和端口

# 假设你有一个用于测试的图片文件
IMAGE_FILE_PATH = 'test.jpg' # 确保这个路径是正确的

def test_predict_api():
    # 检查图片文件是否存在
    if not os.path.exists(IMAGE_FILE_PATH):
        print(f"错误: 找不到图像文件 '{IMAGE_FILE_PATH}'。请确保文件存在于当前目录下。")
        return

    try:
        with open(IMAGE_FILE_PATH, 'rb') as f:
            files = {'file': (IMAGE_FILE_PATH, f, 'image/jpeg')}
            print(f"正在向 {FLASK_APP_URL} 发送请求，文件: {IMAGE_FILE_PATH}")
            response = requests.post(FLASK_APP_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            print("API 调用成功！")
            print(f"预测结果: {json.dumps(result, indent=4, ensure_ascii=False)}")
        else:
            print(f"API 调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"无法连接到服务器。请确保Flask应用正在运行且URL正确。错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    test_predict_api()
