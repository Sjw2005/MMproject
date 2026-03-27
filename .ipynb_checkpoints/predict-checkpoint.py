from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

model = RTDETR(r"runs/detect/train/weights/best.pt")  #使用训练好的模型
#可视化预测结果
model.predict(
    source=r"datasets/images/val", # 测试图像路径，指定可见光图像，会自动读取红外图像
    save=True
)