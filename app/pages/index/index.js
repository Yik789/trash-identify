// pages/index/index.js
Page({
  data: {
    imagePath: '',
    isLoading: false,
    predictionResult: null,
    confidencePercentage: '0.0',
    // 分类映射表
    // 这里我们将类别名直接映射到英文标识，方便CSS类名使用
    categories: {
      '厨余垃圾': { en: 'kitchen', color: '#8BC34A' },
      '可回收物': { en: 'recyclable', color: '#2196F3' },
      '有害垃圾': { en: 'harmful', color: '#F44336' },
      '其他垃圾': { en: 'other', color: '#9E9E9E' }
    }
  },

  onLoad() {
    // 初始化数据，确保所有变量都有定义
    this.setData({
      predictionResult: null,
      confidencePercentage: '0.0'
    });
  },

  // 选择图片
  chooseImage() {
    if (this.data.isLoading) return; // 如果正在加载，则不响应点击

    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'], // 可以选择原图或压缩图，这里选择压缩图
      sourceType: ['album', 'camera'], // 从相册或摄像头选择
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];
        this.setData({ 
          imagePath: tempFilePath,
          predictionResult: null, // 清空上次结果
          confidencePercentage: '0.0'
        });
        // 选择了图片后立即开始识别
        this.recognizeImage(tempFilePath);
      },
      fail: (err) => {
        console.error('选择图片失败:', err);
        wx.showToast({
          title: '选择图片失败',
          icon: 'none'
        });
      }
    });
  },

  // 识别图片 - 现在是真正的API调用
  recognizeImage(filePath) {
    this.setData({ isLoading: true });
    
    // === 替换为您的Flask API地址 ===
    // 注意：如果是正式部署，这里必须是 HTTPS 地址
    const uploadUrl = 'http://100.2.149.36:5000/predict'; 

    wx.uploadFile({
      url: uploadUrl,
      filePath: filePath,
      name: 'file', // 必须与Flask后端 request.files['file'] 匹配
      header: {
        'Content-Type': 'multipart/form-data' // 确保正确的文件上传类型
      },
      success: (res) => {
        // Flask 返回的是 JSON 字符串，需要解析
        // 注意：res.data 是字符串，不是直接的JS对象
        const data = JSON.parse(res.data); 
        
        if (data.error) {
          console.error('API 返回错误:', data.error);
          wx.showToast({
            title: `识别失败: ${data.error}`,
            icon: 'none',
            duration: 2000
          });
        } else {
          console.log('API 预测成功:', data);
          this.handleImageRecognition(data);
        }
      },
      fail: (err) => {
        console.error('上传或预测失败:', err);
        wx.showToast({
          title: '网络或识别失败，请检查后端服务',
          icon: 'none',
          duration: 3000
        });
      },
      complete: () => {
        this.setData({ isLoading: false }); // 无论成功失败，都停止加载状态
      }
    });
  },

  // 处理识别结果
  handleImageRecognition(result) {
    // 计算置信度百分比
    const confidence = result.confidence * 100;
    const confidencePercentage = confidence.toFixed(1);
    
    // 获取分类信息，确保有默认值以防API返回未知类别
    const categoryInfo = this.data.categories[result.category] || 
                         this.data.categories['其他垃圾'] || // Fallback to '其他垃圾' if category not found
                         { en: 'other', color: '#9E9E9E' }; // Final fallback
    
    this.setData({
      predictionResult: {
        category: result.category,
        // 这里使用category_en来动态设置CSS类名
        category_en: categoryInfo.en, 
        confidence: result.confidence
      },
      confidencePercentage: confidencePercentage
    });
  },
  
  // 错误处理（通常由wx.showToast等内部处理，此方法可用于全局捕获）
  onError(error) {
    console.error('页面发生错误:', error);
    wx.showToast({
      title: '发生错误，请重试',
      icon: 'none'
    });
    this.setData({ isLoading: false });
  }
});