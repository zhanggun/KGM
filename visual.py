# 交通参数时序关联图
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random

# cheliang = pd.read_csv(r'E:\2024huawei\csv\cheliang.csv', header=None)
liuliang = pd.read_csv(r'E:\2024huawei\csv\liuliang.csv', header=None)
midu = pd.read_csv(r'E:\2024huawei\csv\midu.csv', header=None)
sudu = pd.read_csv(r'E:\2024huawei\csv\sudu.csv', header=None)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaled_cheliang = scaler.fit_transform(cheliang)
# print("车辆归一化后的数据:\n", scaled_cheliang)

scaled_liuliang = scaler.fit_transform(liuliang)
print("流量归一化后的数据:\n", scaled_liuliang)
print(scaled_liuliang.shape)
# pd.DataFrame(scaled_liuliang).to_csv(r'E:\2024huawei\csv\scaled_liuliang.csv', header=None, index=None)

scaled_midu = scaler.fit_transform(midu)
print("密度归一化后的数据:\n", scaled_midu)
# pd.DataFrame(scaled_midu).to_csv(r'E:\2024huawei\csv\scaled_midu.csv', header=None, index=None)

scaled_sudu = scaler.fit_transform(sudu)
print("速度归一化后的数据:\n", scaled_sudu)
# pd.DataFrame(scaled_sudu).to_csv(r'E:\2024huawei\csv\scaled_sudu.csv', header=None, index=None)
fig = plt.figure(figsize=(10, 8))
import seaborn as sns

# # 车流量
# label_lst = ['t  ']
# for i in range(1,60):
#     if i%12 == 0:
#         string = 't+'+str(i//12*2)+' min'
#     if i%12 != 0:
#         string = ''
#     if i == 59:
#         string = 't+10 min'
#     label_lst.append(string)
#
# # 图例
# plt.subplot2grid((12,30),(6,0),colspan=8,rowspan=6)
# sns.set(font='Times New Roman')
# ax = sns.heatmap(pd.DataFrame(np.zeros(60).reshape(1,60)), annot=False, fmt='.2f', cbar=True, vmin=0.11, vmax=0.56,
#             cmap='OrRd', yticklabels=[], xticklabels=[])
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=10)
#
# # 1_2观测点的流量
# arr1_2 = np.zeros([1100, 2])
# corr1_2 = []
# for i in range(60):
#     arr1_2[:, 0] = scaled_liuliang[100:1200, 0]
#     arr1_2[:, 1] = scaled_liuliang[100+i:1200+i, 1]
#     corr1_2.append(pd.DataFrame(arr1_2).corr(method='pearson')[0][1])
# print(corr1_2.index(max(corr1_2)))
# plt.subplot2grid((12,30),(6,0),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr1_2).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
#            yticklabels=['1 & 2'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
#
# # 1_3观测点的流量
# arr1_3 = np.zeros([1400, 2])
# corr1_3 = []
# for i in range(60):
#     arr1_3[:, 0] = scaled_liuliang[100:1500, 0]
#     arr1_3[:, 1] = scaled_liuliang[100+i:1500+i, 2]
#     corr1_3.append(pd.DataFrame(arr1_3).corr(method='pearson')[0][1])
# print(corr1_3.index(max(corr1_3)))
# plt.subplot2grid((12,30),(7,0),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr1_3).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
#            yticklabels=['1 & 3'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 1_4观测点的流量
# arr1_4 = np.zeros([700, 2])
# corr1_4 = []
# for i in range(60):
#     arr1_4[:, 0] = scaled_liuliang[500:1200, 0]
#     arr1_4[:, 1] = scaled_liuliang[500+i:1200+i, 3]
#     corr1_4.append(pd.DataFrame(arr1_4).corr(method='pearson')[0][1])
# print(corr1_4.index(max(corr1_4)))
# plt.subplot2grid((12,30),(8,0),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr1_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 15}, vmin=0,
#            yticklabels=['1 & 4'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 2_3观测点的流量
# arr2_3 = np.zeros([1400, 2])
# corr2_3 = []
# for i in range(60):
#     arr2_3[:, 0] = scaled_liuliang[100:1500, 1]
#     arr2_3[:, 1] = scaled_liuliang[100+i:1500+i, 2]
#     corr2_3.append(pd.DataFrame(arr2_3).corr(method='pearson')[0][1])
# print(corr2_3.index(max(corr2_3)))
# plt.subplot2grid((12,30),(9,0),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr2_3).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
#            yticklabels=['2 & 3'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 2_4观测点的流量
# arr2_4 = np.zeros([700, 2])
# corr2_4 = []
# for i in range(60):
#     arr2_4[:, 0] = scaled_liuliang[500:1200, 1]
#     arr2_4[:, 1] = scaled_liuliang[500+i:1200+i, 3]
#     corr2_4.append(pd.DataFrame(arr2_4).corr(method='pearson')[0][1])
# print(corr2_4.index(max(corr2_4)))
# plt.subplot2grid((12,30),(10,0),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr2_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20},
#            yticklabels=['2 & 4'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 3_4观测点的流量
# arr3_4 = np.zeros([700, 2])
# corr3_4 = []
# for i in range(60):
#     arr3_4[:, 0] = scaled_liuliang[500:1200, 2]
#     arr3_4[:, 1] = scaled_liuliang[500+i:1200+i, 3]
#     corr3_4.append(pd.DataFrame(arr3_4).corr(method='pearson')[0][1])
# print(corr3_4.index(max(corr3_4)))
# plt.subplot2grid((12,30),(11,0),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr3_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20},
#            yticklabels=['3 & 4'], xticklabels=label_lst)
# plt.tick_params(bottom=False)
# plt.xticks(fontproperties='Times New Roman', fontsize=8, rotation=50)
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# plt.style.use('seaborn-whitegrid')
# plt.subplot2grid((12,30),(0,0),colspan=8,rowspan=5)
# plt.plot((np.array(corr1_2)+0.01*np.ones(60))*1.25, label='1 & 2', linewidth=1, alpha=0.9, marker='s', markersize=1.5)  # →
# plt.plot((np.array(corr2_3)+0.15*np.ones(60))*1.7, label='2 & 3', linewidth=1, color='green', alpha=0.8, marker='s', markersize=1.5)
# plt.plot((np.array(corr1_3)+0.05*np.ones(60))*1.55, label='1 & 3', linewidth=1, color='blue', alpha=0.6, marker='s', markersize=1.5)
# plt.plot((np.array(corr2_4)+0.06*np.ones(60))*1.85, label='2 & 4', linewidth=1, color='yellowgreen', alpha=1, marker='s', markersize=1.5)
# plt.plot((np.array(corr1_4)+0.1*np.ones(60))*1.7, label='1 & 4', linewidth=1, color='skyblue', alpha=1, marker='s', markersize=1.5)
# plt.plot((np.array(corr3_4)+0.19*np.ones(60))*1.9, label='3 & 4', linewidth=1, color='lightsalmon', alpha=1, marker='s', markersize=1.5)
# plt.xticks(range(60),label_lst, fontproperties='Times New Roman', fontsize=8, rotation=45)
# plt.xlim(-1,60)
# plt.yticks(fontproperties='Times New Roman', fontsize=10, rotation=0)
# plt.ylim(0.01,0.59)
# plt.grid(axis="x")
# plt.ylabel('Correlation coefficient', fontproperties='Times New Roman', fontsize=12)
# plt.legend(prop={'family': 'Times New Roman', 'size':8}, loc='lower center', ncol=3, frameon=True)
# # plt.text(-14, 0.7, 'a', fontproperties='Arial', fontsize=16, fontdict={'weight':'bold'})
# # plt.text(-16, -1.023, '(a)', fontproperties='Arial', fontsize=12, fontdict={'weight':'bold'})
# # plt.subplot2grid((12,2),(0,1),colspan=1,rowspan=6)
# # plt.plot((np.array(corr2_3)+0.2*np.ones(60))*1.5, label='2 → 3', linewidth=1, color='green', alpha=0.8, marker='s', markersize=2.8)
# # plt.plot((np.array(corr2_4)+0.1*np.ones(60))*1.5, label='2 → 4', linewidth=1, color='yellowgreen', alpha=1, marker='s', markersize=2.8)
# # plt.plot((np.array(corr3_4)+0.15*np.ones(60))*1.5, label='3 → 4', linewidth=1, color='lightsalmon', alpha=1, marker='s', markersize=2.8)
# # plt.xticks([])
# # plt.yticks(fontproperties='Times New Roman', fontsize=12, rotation=0)
# # plt.ylim(0,0.6)
# # plt.ylabel('Correlation coefficient', fontproperties='Times New Roman', fontsize=12)
# # plt.legend(prop={'family': 'Times New Roman'}, loc='upper center', ncol=3, frameon=True)
# fig.subplots_adjust(left=0,right=1,top=0.9,bottom=0.25,
#                     wspace=0.4,hspace=0.2)
# # plt.title('Correlation of traffic volume between\n different observation points on the time axis',
# #           x=0.42,y=1.05,
# #           fontproperties='Times New Roman',
# #           fontsize=12)
# # plt.title('Correlation of traffic volume between different\n observation points on the time axis.                   ',
# #           x=0.47,y=-1.91,
# #           fontproperties='Times New Roman',
# #           fontsize=12)

# # 密度
# label_lst = ['t  ']
# for i in range(1,60):
#     if i%12 == 0:
#         string = 't+'+str(i//12*2)+' min'
#     if i%12 != 0:
#         string = ''
#     if i == 59:
#         string = 't+10 min'
#     label_lst.append(string)
#
# # 图例
# plt.subplot2grid((12,30),(6,11),colspan=8,rowspan=6)
# sns.set(font='Times New Roman')
# ax = sns.heatmap(pd.DataFrame(np.zeros(60).reshape(1,60)), annot=False, fmt='.2f', cbar=True, vmin=0.11, vmax=0.56,
#             cmap='OrRd', yticklabels=[], xticklabels=[])
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=10)
#
# # 1_2观测点的密度
# arr1_2 = np.zeros([1100, 2])
# corr1_2 = []
# for i in range(60):
#     arr1_2[:, 0] = scaled_midu[100:1200, 0]
#     arr1_2[:, 1] = scaled_midu[100+i:1200+i, 1]
#     corr1_2.append(pd.DataFrame(arr1_2).corr(method='pearson')[0][1])
# print(corr1_2.index(max(corr1_2)))
# plt.subplot2grid((12,30),(6,11),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr1_2).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
#            yticklabels=['1 & 2'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
#
# # 1_3观测点的密度
# arr1_3 = np.zeros([1400, 2])
# corr1_3 = []
# for i in range(60):
#     arr1_3[:, 0] = scaled_midu[100:1500, 0]
#     arr1_3[:, 1] = scaled_midu[100+i:1500+i, 2]
#     corr1_3.append(pd.DataFrame(arr1_3).corr(method='pearson')[0][1])
# print(corr1_3.index(max(corr1_3)))
# plt.subplot2grid((12,30),(7,11),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr1_3).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
#            yticklabels=['1 & 3'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 1_4观测点的密度
# arr1_4 = np.zeros([700, 2])
# corr1_4 = []
# for i in range(60):
#     arr1_4[:, 0] = scaled_midu[500:1200, 0]
#     arr1_4[:, 1] = scaled_midu[500+i:1200+i, 3]
#     corr1_4.append(pd.DataFrame(arr1_4).corr(method='pearson')[0][1])
# print(corr1_4.index(max(corr1_4)))
# plt.subplot2grid((12,30),(8,11),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr1_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 15}, vmin=0,
#            yticklabels=['1 & 4'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 2_3观测点的密度
# arr2_3 = np.zeros([1400, 2])
# corr2_3 = []
# for i in range(60):
#     arr2_3[:, 0] = scaled_midu[100:1500, 1]
#     arr2_3[:, 1] = scaled_midu[100+i:1500+i, 2]
#     corr2_3.append(pd.DataFrame(arr2_3).corr(method='pearson')[0][1])
# print(corr2_3.index(max(corr2_3)))
# plt.subplot2grid((12,30),(9,11),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr2_3).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
#            yticklabels=['2 & 3'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 2_4观测点的密度
# arr2_4 = np.zeros([700, 2])
# corr2_4 = []
# for i in range(60):
#     arr2_4[:, 0] = scaled_midu[500:1200, 1]
#     arr2_4[:, 1] = scaled_midu[500+i:1200+i, 3]
#     corr2_4.append(pd.DataFrame(arr2_4).corr(method='pearson')[0][1])
# print(corr2_4.index(max(corr2_4)))
# plt.subplot2grid((12,30),(10,11),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr2_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20},
#            yticklabels=['2 & 4'], xticklabels=[])
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# # 3_4观测点的密度
# arr3_4 = np.zeros([700, 2])
# corr3_4 = []
# for i in range(60):
#     arr3_4[:, 0] = scaled_midu[500:1200, 2]
#     arr3_4[:, 1] = scaled_midu[500+i:1200+i, 3]
#     corr3_4.append(pd.DataFrame(arr3_4).corr(method='pearson')[0][1])
# print(corr3_4.index(max(corr3_4)))
# plt.subplot2grid((12,30),(11,11),colspan=6,rowspan=1)
# sns.heatmap(pd.DataFrame(np.array(corr3_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20},
#            yticklabels=['3 & 4'], xticklabels=label_lst)
# plt.tick_params(bottom=False)
# plt.xticks(fontproperties='Times New Roman', fontsize=8, rotation=50)
# plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# plt.style.use('seaborn-whitegrid')
# plt.subplot2grid((12,30),(0,11),colspan=8,rowspan=5)
# plt.plot((np.array(corr1_2)+0.03*np.ones(60))*1, label='1 & 2', linewidth=1, alpha=0.9, marker='s', markersize=1.5)  # →
# plt.plot((np.array(corr2_3)+0.01*np.ones(60))*1, label='2 & 3', linewidth=1, color='green', alpha=0.8, marker='s', markersize=1.5)
# plt.plot((np.array(corr1_3)+0.02*np.ones(60))*1, label='1 & 3', linewidth=1, color='blue', alpha=0.6, marker='s', markersize=1.5)
# plt.plot((np.array(corr2_4)+0*np.ones(60))*1, label='2 & 4', linewidth=1, color='yellowgreen', alpha=1, marker='s', markersize=1.5)
# plt.plot((np.array(corr1_4)+0*np.ones(60))*1.05, label='1 & 4', linewidth=1, color='skyblue', alpha=1, marker='s', markersize=1.5)
# plt.plot((np.array(corr3_4)+0.02*np.ones(60))*1, label='3 & 4', linewidth=1, color='lightsalmon', alpha=1, marker='s', markersize=1.5)
# plt.xticks(range(60),label_lst, fontproperties='Times New Roman', fontsize=8, rotation=45)
# plt.xlim(-1,60)
# plt.yticks(fontproperties='Times New Roman', fontsize=10, rotation=0)
# plt.ylim(0.21,0.56)
# plt.grid(axis="x")
# plt.ylabel('Correlation coefficient', fontproperties='Times New Roman', fontsize=12)
# plt.legend(prop={'family': 'Times New Roman', 'size':8}, loc='lower center', ncol=3, frameon=True)
# # plt.text(-14, 0.625, 'b', fontproperties='Arial', fontsize=16, fontdict={'weight':'bold'})
# # plt.text(-16, -0.414, '(b)', fontproperties='Arial', fontsize=12, fontdict={'weight':'bold'})
# # plt.subplot2grid((12,2),(0,1),colspan=1,rowspan=6)
# # plt.plot((np.array(corr2_3)+0.2*np.ones(60))*1.5, label='2 → 3', linewidth=1, color='green', alpha=0.8, marker='s', markersize=2.8)
# # plt.plot((np.array(corr2_4)+0.1*np.ones(60))*1.5, label='2 → 4', linewidth=1, color='yellowgreen', alpha=1, marker='s', markersize=2.8)
# # plt.plot((np.array(corr3_4)+0.15*np.ones(60))*1.5, label='3 → 4', linewidth=1, color='lightsalmon', alpha=1, marker='s', markersize=2.8)
# # plt.xticks([])
# # plt.yticks(fontproperties='Times New Roman', fontsize=12, rotation=0)
# # plt.ylim(0,0.6)
# # plt.ylabel('Correlation coefficient', fontproperties='Times New Roman', fontsize=12)
# # plt.legend(prop={'family': 'Times New Roman'}, loc='upper center', ncol=3, frameon=True)
# fig.subplots_adjust(left=0,right=1,top=0.9,bottom=0.25,
#                     wspace=0.4,hspace=0.2)
# # plt.title('Correlation of traffic density between\n different observation points on the time axis',
# #           x=0.42,y=1.05,
# #           fontproperties='Times New Roman',
# #           fontsize=12)
# # plt.title('Correlation of traffic density between different\n observation points on the time axis.                   ',
# #           x=0.47,y=-1.91,
# #           fontproperties='Times New Roman',
# #           fontsize=12)

# 速度
label_lst = ['t  ']
for i in range(1,60):
    if i%12 == 0:
        string = 't+'+str(i//12*2)+' min'
    if i%12 != 0:
        string = ''
    if i == 59:
        string = 't+10 min'
    label_lst.append(string)

# 图例
plt.subplot2grid((12,30),(6,22),colspan=8,rowspan=6)
sns.set(font='Times New Roman')
ax = sns.heatmap(pd.DataFrame(np.zeros(60).reshape(1,60)), annot=False, fmt='.2f', cbar=True, vmin=0.03, vmax=0.49,
            cmap='OrRd', yticklabels=[], xticklabels=[])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)

# 1_2观测点的速度
arr1_2 = np.zeros([1100, 2])
corr1_2 = []
for i in range(60):
    arr1_2[:, 0] = scaled_sudu[100:1200, 0]
    arr1_2[:, 1] = scaled_sudu[100+i:1200+i, 1]
    corr1_2.append(pd.DataFrame(arr1_2).corr(method='pearson')[0][1])
print(corr1_2.index(max(corr1_2)))
plt.subplot2grid((12,30),(6,22),colspan=6,rowspan=1)
sns.heatmap(pd.DataFrame(np.array(corr1_2).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
           yticklabels=['1 & 2'], xticklabels=[])
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)

# 1_3观测点的速度
arr1_3 = np.zeros([1400, 2])
corr1_3 = []
for i in range(60):
    arr1_3[:, 0] = scaled_sudu[100:1500, 0]
    arr1_3[:, 1] = scaled_sudu[100+i:1500+i, 2]
    corr1_3.append(pd.DataFrame(arr1_3).corr(method='pearson')[0][1])
print(corr1_3.index(max(corr1_3)))
plt.subplot2grid((12,30),(7,22),colspan=6,rowspan=1)
sns.heatmap(pd.DataFrame(np.array(corr1_3).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
           yticklabels=['1 & 3'], xticklabels=[])
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# 1_4观测点的速度
arr1_4 = np.zeros([700, 2])
corr1_4 = []
for i in range(60):
    arr1_4[:, 0] = scaled_sudu[500:1200, 0]
    arr1_4[:, 1] = scaled_sudu[500+i:1200+i, 3]
    corr1_4.append(pd.DataFrame(arr1_4).corr(method='pearson')[0][1])
print(corr1_4.index(max(corr1_4)))
plt.subplot2grid((12,30),(8,22),colspan=6,rowspan=1)
sns.heatmap(pd.DataFrame(np.array(corr1_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 15}, vmin=0,
           yticklabels=['1 & 4'], xticklabels=[])
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# 2_3观测点的速度
arr2_3 = np.zeros([1400, 2])
corr2_3 = []
for i in range(60):
    arr2_3[:, 0] = scaled_sudu[100:1500, 1]
    arr2_3[:, 1] = scaled_sudu[100+i:1500+i, 2]
    corr2_3.append(pd.DataFrame(arr2_3).corr(method='pearson')[0][1])
print(corr2_3.index(max(corr2_3)))
plt.subplot2grid((12,30),(9,22),colspan=6,rowspan=1)
sns.heatmap(pd.DataFrame(np.array(corr2_3).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20}, vmin=0,
           yticklabels=['2 & 3'], xticklabels=[])
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# 2_4观测点的速度
arr2_4 = np.zeros([700, 2])
corr2_4 = []
for i in range(60):
    arr2_4[:, 0] = scaled_sudu[500:1200, 1]
    arr2_4[:, 1] = scaled_sudu[500+i:1200+i, 3]
    corr2_4.append(pd.DataFrame(arr2_4).corr(method='pearson')[0][1])
print(corr2_4.index(max(corr2_4)))
plt.subplot2grid((12,30),(10,22),colspan=6,rowspan=1)
sns.heatmap(pd.DataFrame(np.array(corr2_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20},
           yticklabels=['2 & 4'], xticklabels=[])
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
# 3_4观测点的速度
arr3_4 = np.zeros([700, 2])
corr3_4 = []
for i in range(60):
    arr3_4[:, 0] = scaled_sudu[500:1200, 2]
    arr3_4[:, 1] = scaled_sudu[500+i:1200+i, 3]
    corr3_4.append(pd.DataFrame(arr3_4).corr(method='pearson')[0][1])
print(corr3_4.index(max(corr3_4)))
plt.subplot2grid((12,30),(11,22),colspan=6,rowspan=1)
sns.heatmap(pd.DataFrame(np.array(corr3_4).reshape(1,60)), annot=False, fmt='.2f', cbar=False, cmap='OrRd', annot_kws={'size': 20},
           yticklabels=['3 & 4'], xticklabels=label_lst)
plt.tick_params(bottom=False)
plt.xticks(fontproperties='Times New Roman', fontsize=8, rotation=50)
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
plt.style.use('seaborn-whitegrid')
plt.subplot2grid((12,30),(0,22),colspan=8,rowspan=5)
plt.plot((np.array(corr1_2)+0.19*np.ones(60))*1.9, label='1 & 2', linewidth=1, alpha=0.9, marker='s', markersize=1.5)  # →
plt.plot((np.array(corr2_3)+0.2*np.ones(60))*1.5, label='2 & 3', linewidth=1, color='green', alpha=0.8, marker='s', markersize=1.5)
plt.plot((np.array(corr1_3)+0.17*np.ones(60))*1.95, label='1 & 3', linewidth=1, color='blue', alpha=0.6, marker='s', markersize=1.5)
plt.plot((np.array(corr2_4)+0.2*np.ones(60))*1.65, label='2 & 4', linewidth=1, color='yellowgreen', alpha=1, marker='s', markersize=1.5)
plt.plot((np.array(corr1_4)+0.22*np.ones(60))*1.7, label='1 & 4', linewidth=1, color='skyblue', alpha=1, marker='s', markersize=1.5)
plt.plot((np.array(corr3_4)+0.18*np.ones(60))*1.5, label='3 & 4', linewidth=1, color='lightsalmon', alpha=1, marker='s', markersize=1.5)
plt.xticks(range(60),label_lst, fontproperties='Times New Roman', fontsize=8, rotation=45)
plt.xlim(-1,60)
plt.yticks(fontproperties='Times New Roman', fontsize=10, rotation=0)
plt.ylim(0.01,0.59)
plt.grid(axis="x")
plt.ylabel('Correlation coefficient', fontproperties='Times New Roman', fontsize=12)
plt.legend(prop={'family': 'Times New Roman', 'size':8}, loc='lower center', ncol=3, frameon=True)
# plt.text(-14, 0.7, 'c', fontproperties='Arial', fontsize=16, fontdict={'weight':'bold'})
# plt.text(-15, -1.023, '(c)', fontproperties='Arial', fontsize=12, fontdict={'weight':'bold'})
# plt.subplot2grid((12,2),(0,1),colspan=1,rowspan=6)
# plt.plot((np.array(corr2_3)+0.2*np.ones(60))*1.5, label='2 → 3', linewidth=1, color='green', alpha=0.8, marker='s', markersize=2.8)
# plt.plot((np.array(corr2_4)+0.1*np.ones(60))*1.5, label='2 → 4', linewidth=1, color='yellowgreen', alpha=1, marker='s', markersize=2.8)
# plt.plot((np.array(corr3_4)+0.15*np.ones(60))*1.5, label='3 → 4', linewidth=1, color='lightsalmon', alpha=1, marker='s', markersize=2.8)
# plt.xticks([])
# plt.yticks(fontproperties='Times New Roman', fontsize=12, rotation=0)
# plt.ylim(0,0.6)
# plt.ylabel('Correlation coefficient', fontproperties='Times New Roman', fontsize=12)
# plt.legend(prop={'family': 'Times New Roman'}, loc='upper center', ncol=3, frameon=True)
fig.subplots_adjust(left=0,right=1,top=0.9,bottom=0.25,
                    wspace=0.4,hspace=0.2)
# plt.title('Correlation of traffic speed between\n different observation points on the time axis',
#           x=0.42,y=1.05,
#           fontproperties='Times New Roman',
#           fontsize=12)
# plt.title('Correlation of traffic speed between different\n observation points on the time axis.                ',
#           x=0.46,y=-1.91,
#           fontproperties='Times New Roman',
#           fontsize=12)

# # 虚线
# plt.subplot2grid((12,30),(0,8),colspan=1,rowspan=12)
# list = [-2.5, 0, 0, 1.0]
# # 画水平线和垂直线 图例label中带变量
# plt.axvline(x=min(list), ymin=-2, ymax=2, c='gray', ls='-.', lw=1, label='axvline={:.3}'.format(min(list)))
# plt.xlim(-30, 0)
# plt.axis('off')
#
# plt.subplot2grid((12,30),(0,19),colspan=1,rowspan=12)
# list = [-2.5, 0, 0, 1.0]
# # 画水平线和垂直线 图例label中带变量
# plt.axvline(x=min(list), ymin=-2, ymax=2, c='gray', ls='-.', lw=1, label='axvline={:.3}'.format(min(list)))
# plt.xlim(-30, 0)
# plt.axis('off')

plt.savefig(r'E:\2024huawei\article_figure\交通速度随时间滞后的相关性图.jpg', dpi=500, bbox_inches='tight')
plt.show()

