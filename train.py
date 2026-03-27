from ultralytics import RTDETR,YOLO
from ultralytics.nn import AddModules


import warnings
warnings.filterwarnings('ignore')
# 实时保存日志，断网可训↓↓↓  
# 终端运行命令：nohup python train.py > logs/DMFNet.txt 2>&1 & echo $! > logs/train.pid && tail -f logs/DMFNet.txt

model = YOLO("./improve_multimodal/DMFNet.yaml")


model.train(
    # resume=True,#断点续训
    # pretrained=True,#追加训练
    data='data.yaml',  # 训练参数均可以重新设置
    epochs=150,
    imgsz=640,  #修改
    workers=8,
    batch=16,
    device=0,
    cache="ram",
    optimizer='SGD',
    amp=False,
    project="results",
    name="DMFNet",    #保存到该文件夹下
#Add
    warmup_epochs=5.0, 
    mosaic=1.0,
    close_mosaic=30,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    lr0=0.01,
    scale=0.2,
    max_det=500,
    conf=0.001,
    hsv_h=0.01,
    hsv_s=0.01,
    hsv_v=0.01

)
print("✅  模型训练完成！")

