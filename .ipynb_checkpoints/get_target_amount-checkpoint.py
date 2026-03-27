import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,                 # 全局基础字号
    "axes.labelsize": 10,           # 坐标轴标签字号
    "axes.titlesize": 10,           # 标题字号
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,          # 坐标轴线宽
    "xtick.major.width": 0.8,       # 刻度线宽
    "ytick.major.width": 0.8,
    "xtick.direction": "in",        # 刻度朝内
    "ytick.direction": "in",
    "legend.frameon": False,        # 图例无边框
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False
})

def getGtAreaAndRatio(label_dir):
    data_dict = {}
    assert Path(label_dir).is_dir(), f"label_dir is not exist: {label_dir}"

    txts = os.listdir(label_dir)

    for txt in txts:
        if not txt.endswith(".txt"):
            continue

        txt_path = os.path.join(label_dir, txt)
        if not os.path.isfile(txt_path):
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue

            temp = line.split()
            if len(temp) < 5:
                print(f"警告: 文件 {txt} 的第 {line_num} 行格式错误，数据不足: {line}")
                continue

            try:
                # 只取前4个坐标值 (x, y, w, h)，这里假设为归一化坐标
                coor_list = list(map(float, temp[1:5]))
                area = coor_list[2] * coor_list[3]  # 归一化面积
                ratio = round(coor_list[2] / coor_list[3], 2) if coor_list[3] != 0 else 0

                cls_id = temp[0]
                if cls_id not in data_dict:
                    data_dict[cls_id] = {'area': [], 'ratio': []}

                data_dict[cls_id]['area'].append(area)
                data_dict[cls_id]['ratio'].append(ratio)

            except ValueError as e:
                print(f"警告: 文件 {txt} 的第 {line_num} 行包含非数字数据: {line}, 错误: {e}")
            except ZeroDivisionError:
                print(f"警告: 文件 {txt} 的第 {line_num} 行宽高比计算出现除以零: {line}")

    return data_dict


def getSMLGtNumByClass(data_dict, class_num):
    """
    计算某个类别的小物体、中物体、大物体的个数
    params class_num: 类别  0, 1, 2, ...
    return s: 小物体个数  0 < area <= 0.5%
           m: 中物体个数  0.5% < area <= 1%
           l: 大物体个数  area > 1%
    """
    s, m, l = 0, 0, 0
    # ✅️图片的尺寸大小 注意修改!!!  e.g. 1280*1024   640*512
    h = 512
    w = 640

    class_key = str(class_num)
    if class_key not in data_dict:
        print(f"警告: 类别 {class_num} 不存在于数据字典中")
        return s, m, l

    for item in data_dict[class_key]['area']:
        pixel_area = item * h * w
        if pixel_area <= h * w * 0.005:
            s += 1
        elif pixel_area <= h * w * 0.010:
            m += 1
        else:
            l += 1
    return s, m, l


def getAllSMLGtNum(data_dict, isEachClass=False):    #数据集所有类别小、中、大分布情况
    S, M, L = 0, 0, 0

    classDict = {cls: {'S': 0, 'M': 0, 'L': 0} for cls in data_dict.keys()}

    if not isEachClass:
        for class_num in data_dict.keys():
            s, m, l = getSMLGtNumByClass(data_dict, class_num)
            S += s
            M += m
            L += l
        return [S, M, L]
    else:
        for class_num in data_dict.keys():
            s, m, l = getSMLGtNumByClass(data_dict, class_num)
            classDict[class_num]['S'] = s
            classDict[class_num]['M'] = m
            classDict[class_num]['L'] = l
        return classDict


def plotAllSML(SML, save_dir, filename="object_size_distribution.svg"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    colors = ['#4C72B0', '#55A868', '#C44E52']  # blue / green / red-ish

    # 单栏图常用尺寸
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    categories = ['Small', 'Medium', 'Large']
    bars = ax.bar(categories, SML,
                  color=colors, width=0.6,
                  edgecolor='black', linewidth=0.8)

    y_max = max(SML) * 1.15 if max(SML) > 0 else 1
    ax.set_ylim(0, y_max)

    # 数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + y_max * 0.02,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=8)

    ax.set_xlabel('Object Size Category', fontsize=10)
    ax.set_ylabel('Number of Objects', fontsize=10)
    ax.set_title('Size Distribution of Objects in Drone Dataset',  #✅️图片上方标签
                 fontsize=10, pad=6)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close(fig)


def getClassCount(data_dict):   #统计每个类别的目标数量
    class_count_dict = {}
    for cls, info in data_dict.items():
        class_count_dict[cls] = len(info['area'])
    return class_count_dict


def plotClassDistribution(class_count_dict,
                          save_dir,
                          filename="class_distribution.svg",
                          class_name_map=None):

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 按类别 ID 排序
    class_ids = sorted(class_count_dict.keys(), key=lambda x: int(x))
    counts = [class_count_dict[cls] for cls in class_ids]

    if class_name_map is not None:
        x_labels = [class_name_map.get(cls, cls) for cls in class_ids]
    else:
        x_labels = class_ids

    num_classes = len(class_ids)

    base_colors = ['#4C72B0', '#55A868', '#C44E52',
                   '#8172B3', '#CCB974', '#64B5CD']
    if num_classes <= len(base_colors):
        colors = base_colors[:num_classes]
    else:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(num_classes)]

    width_per_class = 0.18
    fig_width = max(3.5, min(7.0, width_per_class * num_classes + 1.5))
    fig_height = 2.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(num_classes)
    bars = ax.bar(x, counts,
                  color=colors,
                  width=0.7,
                  edgecolor='black',
                  linewidth=0.8)

    y_max = max(counts) * 1.15 if max(counts) > 0 else 1
    ax.set_ylim(0, y_max)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + y_max * 0.02,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=8)

    ax.set_xlabel('Object Category', fontsize=10)
    ax.set_ylabel('Number of Instances', fontsize=10)
    ax.set_title('Category-wise Object Statistics in Drone Dataset',  #✅️图片上方标签
                 fontsize=10, pad=6)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')

    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close(fig)


if __name__ == '__main__':
    # labels 根目录，下面有 train、val、test 等子文件夹
    label_root = r"D:\毕设\数据集\DroneVehicle\labels"  # ✅️设置路径
    save_dir = r"C:\Users\25783\Desktop\论文-数据"

    total_SML = [0, 0, 0]  # [S_sum, M_sum, L_sum]
    total_class_count = defaultdict(int)
    total_class_SML = defaultdict(lambda: {'S': 0, 'M': 0, 'L': 0})

    for sub in sorted(os.listdir(label_root)):
        sub_dir = os.path.join(label_root, sub)
        if not os.path.isdir(sub_dir):
            continue

        data_dict = getGtAreaAndRatio(sub_dir)

        if not data_dict:
            print(f"{sub} 文件夹为空或无有效标签，跳过。")
            continue

        SML = getAllSMLGtNum(data_dict, isEachClass=False)
        total_SML[0] += SML[0]
        total_SML[1] += SML[1]
        total_SML[2] += SML[2]

        class_count_dict = getClassCount(data_dict)
        for cls, cnt in class_count_dict.items():
            total_class_count[cls] += cnt

        class_SML_dict = getAllSMLGtNum(data_dict, isEachClass=True)
        for cls, sml_dict in class_SML_dict.items():
            total_class_SML[cls]['S'] += sml_dict['S']
            total_class_SML[cls]['M'] += sml_dict['M']
            total_class_SML[cls]['L'] += sml_dict['L']

        print(f"{sub}文件夹：")
        print(f"小物体: {SML[0]}个, 中物体: {SML[1]}个, 大物体: {SML[2]}个")

        ordered_class_count = {
            cls: class_count_dict[cls]
            for cls in sorted(class_count_dict.keys(), key=lambda x: int(x))
        }
        print(f"各类别目标数量： {ordered_class_count}")
        print("各类别小/中/大目标数量：")
        for cls in sorted(class_SML_dict.keys(), key=lambda x: int(x)):
            s = class_SML_dict[cls]['S']
            m = class_SML_dict[cls]['M']
            l = class_SML_dict[cls]['L']
            print(f"  类别 {cls}: 小={s}, 中={m}, 大={l}")
        print("-" * 60)

    # ✅ 类别标注
    class_name_map = {
        # "0": "Person",
        "0": "Car",
        "1": "Bus",
        "2": "Van",
        "3": "Truck",
        "4": "Freight_Car",
    }

    plotAllSML(total_SML, save_dir, filename="object_size_distribution.svg")

    total_class_count_dict = dict(total_class_count)
    ordered_total_class_count = {
        cls: total_class_count_dict[cls]
        for cls in sorted(total_class_count_dict.keys(), key=lambda x: int(x))
    }
    print("所有文件夹汇总 - 各类别目标数量：", ordered_total_class_count)
    print("所有文件夹汇总 - 各类别小/中/大目标数量：")
    for cls in sorted(total_class_SML.keys(), key=lambda x: int(x)):
        s = total_class_SML[cls]['S']
        m = total_class_SML[cls]['M']
        l = total_class_SML[cls]['L']
        name = class_name_map.get(cls, cls)
        print(f"  类别 {cls} ({name}): 小={s}, 中={m}, 大={l}")

    plotClassDistribution(
        total_class_count_dict,
        save_dir=save_dir,
        filename="class_distribution.svg",
        class_name_map=class_name_map
    )