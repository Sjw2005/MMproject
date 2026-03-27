from ultralytics import RTDETR,YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = RTDETR('./results/resnet18-DEYOLO-try11.12/weights/best.pt')
#输出各项参数
    model.val(data='data.yaml',
                imgsz=640,
#原因参考：www.bilibili.com/video/BV1ZaZhYvE4u/?spm_id_from=333.337.search-card.all.click&vd_source=a53ef52f911c55bbd14dfed2a75d768c
                batch=16,  #设置成train的两倍，否则会出现train、val验证结果不对齐的情况
                split='val',
                workers=8,
                device='0',
                # visualize=True,  #显示过程的特征图
                # save=True,
                )
