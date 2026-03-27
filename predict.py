from ultralytics import RTDETR,YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = RTDETR(r"runs/detect/train/weights/best.pt")  
#可视化预测结果
    model.predict(
        source=r"datasets/images/val", 
        save=True
        
                    )