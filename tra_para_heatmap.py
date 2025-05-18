import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv(r'E:\2024huawei\NGSIM-US-101-trajectory-dataset-smoothing-master\smoothed-dataset\window-11\0820_0835_us101_smoothed_11_\0820_0835_us101_smoothed_11_.csv')

# 提取车道数据
lanedata1 = data[data.Lane_ID == 1]
lanedata2 = data[data.Lane_ID == 2]
lanedata3 = data[data.Lane_ID == 3]
lanedata4 = data[data.Lane_ID == 4]
lanedata5 = data[data.Lane_ID == 5]

# 提取车辆编号
x_vehID1 = lanedata1.drop_duplicates(['Vehicle_ID'])
x_vehID2 = lanedata2.drop_duplicates(['Vehicle_ID'])
x_vehID3 = lanedata3.drop_duplicates(['Vehicle_ID'])
x_vehID4 = lanedata4.drop_duplicates(['Vehicle_ID'])
x_vehID5 = lanedata5.drop_duplicates(['Vehicle_ID'])

# 依据 Global_Time 按照时间先后顺序排序
x_vehID1 = x_vehID1.sort_values(by='Global_Time')
x_vehID2 = x_vehID2.sort_values(by='Global_Time')
x_vehID3 = x_vehID3.sort_values(by='Global_Time')
x_vehID4 = x_vehID4.sort_values(by='Global_Time')
x_vehID5 = x_vehID5.sort_values(by='Global_Time')

# 对排序后的车辆 ID 的索引进行重置，方便索引
x_vehID1 = x_vehID1.reset_index(drop = True)
x_vehID2 = x_vehID2.reset_index(drop = True)
x_vehID3 = x_vehID3.reset_index(drop = True)
x_vehID4 = x_vehID4.reset_index(drop = True)
x_vehID5 = x_vehID5.reset_index(drop = True)

# 导入字体属性相关的包或类
from matplotlib.font_manager import FontProperties
# 预设字体类型、大小
font2 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=10)
#设置画布的尺寸
plt.figure(figsize=(6, 5))

# 1车道
times1 = (len(x_vehID1)-1)  # (len(x_vehID)-1)
i = 0
x1 = []
y1 = []
v1 = []
while i < times1:
    # 循环绘制轨迹图
    lst1 = []
    for j in range(len(lanedata1.Vehicle_ID)):
        if lanedata1['Vehicle_ID'].iloc[j] == x_vehID1.iloc[i][0]:
            lst1.append(j)
    cardata1 = lanedata1.iloc[lst1, :]
#     cardata = lanedata[lanedata.Vehicle_ID == x_vehID[i]]
    # 将时间赋值给变量 x
    x1.extend(list(((cardata1['Global_Time']-10e11)/10e5-1.18848e5)*10))
    # 计算相对移动距离,并赋值给变量 y
    y1.extend(list((np.square(cardata1['Local_Y']) + np.square(cardata1['Local_X']))/10e5))
    # 将速度赋值给变量 v，同时定义速度为颜色映射
    v1.extend(list(cardata1['v_Vel']*2))
    print(i)
    i = i + 1

# # 2车道
# times2 = (len(x_vehID2)-1)  # (len(x_vehID)-1)
# i = 0
# x2 = []
# y2 = []
# v2 = []
# while i < times2:
#     # 循环绘制轨迹图
#     lst2 = []
#     for j in range(len(lanedata2.Vehicle_ID)):
#         if lanedata2['Vehicle_ID'].iloc[j] == x_vehID2.iloc[i][0]:
#             lst2.append(j)
#     cardata2 = lanedata2.iloc[lst2, :]
# #     cardata = lanedata[lanedata.Vehicle_ID == x_vehID[i]]
#     # 将时间赋值给变量 x
#     x2.extend(list(((cardata2['Global_Time']-10e11)/10e5-1.18848e5)*10))
#     # 计算相对移动距离,并赋值给变量 y
#     y2.extend(list((np.square(cardata2['Local_Y']) + np.square(cardata2['Local_X']))/10e5))
#     # 将速度赋值给变量 v，同时定义速度为颜色映射
#     v2.extend(list(cardata2['v_Vel']*2))
#     print(i)
#     i = i + 1

# 3车道
times3 = (len(x_vehID3)-1)  # (len(x_vehID)-1)
i = 0
x3 = []
y3 = []
v3 = []
while i < times3:
    # 循环绘制轨迹图
    lst3 = []
    for j in range(len(lanedata3.Vehicle_ID)):
        if lanedata3['Vehicle_ID'].iloc[j] == x_vehID3.iloc[i][0]:
            lst3.append(j)
    cardata3 = lanedata3.iloc[lst3, :]
#     cardata = lanedata[lanedata.Vehicle_ID == x_vehID[i]]
    # 将时间赋值给变量 x
    x3.extend(list(((cardata3['Global_Time']-10e11)/10e5-1.18848e5)*10))
    # 计算相对移动距离,并赋值给变量 y
    y3.extend(list((np.square(cardata3['Local_Y']) + np.square(cardata3['Local_X']))/10e5))
    # 将速度赋值给变量 v，同时定义速度为颜色映射
    v3.extend(list(cardata3['v_Vel']*2))
    print(i)
    i = i + 1

# 4车道
times4 = (len(x_vehID4)-1)  # (len(x_vehID)-1)
i = 0
x4 = []
y4 = []
v4 = []
while i < times4:
    # 循环绘制轨迹图
    lst4 = []
    for j in range(len(lanedata4.Vehicle_ID)):
        if lanedata4['Vehicle_ID'].iloc[j] == x_vehID4.iloc[i][0]:
            lst4.append(j)
    cardata4 = lanedata4.iloc[lst4, :]
#     cardata = lanedata[lanedata.Vehicle_ID == x_vehID[i]]
    # 将时间赋值给变量 x
    x4.extend(list(((cardata4['Global_Time']-10e11)/10e5-1.18848e5)*10))
    # 计算相对移动距离,并赋值给变量 y
    y4.extend(list((np.square(cardata4['Local_Y']) + np.square(cardata4['Local_X']))/10e5))
    # 将速度赋值给变量 v，同时定义速度为颜色映射
    v4.extend(list(cardata4['v_Vel']*2))
    print(i)
    i = i + 1

# 5车道
times5 = (len(x_vehID5)-1)  # (len(x_vehID)-1)
i = 0
x5 = []
y5 = []
v5 = []
while i < times5:
    # 循环绘制轨迹图
    lst5 = []
    for j in range(len(lanedata5.Vehicle_ID)):
        if lanedata5['Vehicle_ID'].iloc[j] == x_vehID5.iloc[i][0]:
            lst5.append(j)
    cardata5 = lanedata5.iloc[lst5, :]
#     cardata = lanedata[lanedata.Vehicle_ID == x_vehID[i]]
    # 将时间赋值给变量 x
    x5.extend(list(((cardata5['Global_Time']-10e11)/10e5-1.18848e5)*10))
    # 计算相对移动距离,并赋值给变量 y
    y5.extend(list((np.square(cardata5['Local_Y']) + np.square(cardata5['Local_X']))/10e5))
    # 将速度赋值给变量 v，同时定义速度为颜色映射
    v5.extend(list(cardata5['v_Vel']*2))
    print(i)
    i = i + 1

#设定每个图的colormap和colorbar所表示范围是一样的，即归一化
norm = matplotlib.colors.Normalize(vmin=0, vmax=110)
# 绘制散点图
plt.scatter(x5, y5, marker = '.', s=1, c=v5, cmap='jet_r', norm = norm, alpha=0.8)
plt.scatter(x1, y1, marker = '.', s=1, c=v1, cmap='jet_r', norm = norm, alpha=0.8)
# plt.scatter(x2, y2, marker = '.', s=1, c=v2, cmap='jet_r', norm = norm, alpha=0.8)
plt.scatter(x3, y3, marker = '.', s=1, c=v3, cmap='jet_r', norm = norm, alpha=0.8)
plt.scatter(x4, y4, marker = '.', s=1, c=v4, cmap='jet_r', norm = norm, alpha=0.8)
# 设置 X 坐标轴刻度
# plt.text(x, y, '8:05', fontproperties=font2)
# plt.text(x, y, '8:10', fontproperties=font2)
# plt.text(x, y, '8:15', fontproperties=font2)
# plt.text(x, y, '8:20', fontproperties=font2)
plt.xlabel('Time (H:M:S)', fontproperties='Times New Roman', fontsize=12)
plt.ylabel('Distance / km', fontproperties='Times New Roman', fontsize=12)
# plt.title('speed data', fontproperties='Times New Roman', fontsize=15)
# plt.legend(loc='upper left', prop={'family': 'Times New Roman', 'size':12})
plt.xticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
plt.xlim(8.5, 13.5)
plt.ylim(0, 4.5)
scale_x = range(9, 14)
index_x = ['12:10:00', '13:00:00', '13:50:00', '14:40:00', '15:30:00']
plt.xticks(scale_x, index_x)
scale_y = range(0, 5)
index_y = ['0', '1.1', '2.2', '3.3', '4.4']
plt.yticks(scale_y, index_y)
# 添加颜色条
plt.clim(0, 110)
cb = plt.colorbar()
cb.ax.set_title('Km per hour',  fontdict={'family': 'Times New Roman', 'size': 12})
cb.ax.tick_params(labelsize=10)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
plt.savefig(r'E:\2024huawei\article_figure\single_figure\0820_0835_us101_smoothed_11_speed.jpg',dpi=500,bbox_inches='tight')
plt.show()
