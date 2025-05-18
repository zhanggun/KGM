from PyEMD import EMD
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from vmdpy import VMD

liuliang = pd.read_csv(r'E:\2024huawei\csv\liuliang.csv', header=None)
midu = pd.read_csv(r'E:\2024huawei\csv\midu.csv', header=None)
sudu = pd.read_csv(r'E:\2024huawei\csv\sudu.csv', header=None)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaled_cheliang = scaler.fit_transform(cheliang)
# print("车辆归一化后的数据:\n", scaled_cheliang)

scaled_liuliang = scaler.fit_transform(liuliang)
# print("流量归一化后的数据:\n", scaled_liuliang)
print(scaled_liuliang.shape)
# pd.DataFrame(scaled_liuliang).to_csv(r'E:\2024huawei\csv\scaled_liuliang.csv', header=None, index=None)

scaled_midu = scaler.fit_transform(midu)
# print("密度归一化后的数据:\n", scaled_midu)
# pd.DataFrame(scaled_midu).to_csv(r'E:\2024huawei\csv\scaled_midu.csv', header=None, index=None)

scaled_sudu = scaler.fit_transform(sudu)
# print("速度归一化后的数据:\n", scaled_sudu)
# pd.DataFrame(scaled_sudu).to_csv(r'E:\2024huawei\csv\scaled_sudu.csv', header=None, index=None)

scaled_liuliang = np.array(scaled_liuliang)
scaled_liuliang = scaled_liuliang.reshape(scaled_liuliang.shape[1], scaled_liuliang.shape[0])

scaled_midu = np.array(scaled_midu)
scaled_midu = scaled_midu.reshape(scaled_midu.shape[1], scaled_midu.shape[0])

scaled_sudu = np.array(scaled_sudu)
scaled_sudu = scaled_sudu.reshape(scaled_sudu.shape[1], scaled_sudu.shape[0])


sca_ll = np.zeros([1, 6560])
sca_ll[0, :1640] = scaled_liuliang[0, :]
sca_ll[0, 1640*1:1640*2] = scaled_liuliang[1, :]
sca_ll[0, 1640*2:1640*3] = scaled_liuliang[2, :]
sca_ll[0, 1640*3:1640*4] = scaled_liuliang[3, :]

sca_md = np.zeros([1, 6560])
sca_md[0, :1640] = scaled_midu[0, :]
sca_md[0, 1640*1:1640*2] = scaled_midu[1, :]
sca_md[0, 1640*2:1640*3] = scaled_midu[2, :]
sca_md[0, 1640*3:1640*4] = scaled_midu[3, :]

sca_sd = np.zeros([1, 6560])
sca_sd[0, :1640] = scaled_sudu[0, :]
sca_sd[0, 1640*1:1640*2] = scaled_sudu[1, :]
sca_sd[0, 1640*2:1640*3] = scaled_sudu[2, :]
sca_sd[0, 1640*3:1640*4] = scaled_sudu[3, :]

# emd = EMD()
# IMF = emd.emd(liuliang[1, :])
# print(IMF.shape)
# # 绘制 IMF
# N = 8

import pywt
fig= plt.figure(figsize=(11, 3))

# # 第一个
# sr = 128  # 1.sampling rate
# wavename = 'morl'  # 2.母小波名称
# totalscal = 50  # 3.totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
# fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
# cparam = 2 * fc * totalscal  # 常数c
# scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
# [cwtmatr, frequencies] = pywt.cwt(sca_ll[0, :], scales, wavename, 1.0 / sr)  # 4.y为将要进行cwt变换的一维输入信号
# t = np.arange(0, sca_ll[0, :].shape[0]/sr, 1.0/sr)

# t = np.flip(t)
# frequencies = np.flip(frequencies)
# cwtmatr = np.flip(cwtmatr, axis=1)

# ax1 = fig.add_subplot(131, projection='3d')
# print(t.shape)
# print(frequencies.shape)
# print(cwtmatr.shape)
# im1 = ax1.contour3D(t[:3100, ]/5-5.3, frequencies[20:46, ]/10-2.8, abs(cwtmatr[20:46, :3100]), 150, cmap='jet')
# ax1.invert_xaxis()
# ax1.view_init(elev=15, azim=-45)
# ax1.tick_params(labelsize=10, pad=0.02)
# for l in ax1.xaxis.get_ticklabels():
#     l.set_family('Times New Roman')
# for l in ax1.yaxis.get_ticklabels():
#     l.set_family('Times New Roman')
# for l in ax1.zaxis.get_ticklabels():
#     l.set_family('Times New Roman')
# ax1.set_xlabel('Distance [km]', fontproperties='Times New Roman', fontsize=10, rotation=-20)
# ax1.set_ylabel('Time [hour]', fontproperties='Times New Roman', fontsize=10, rotation=60)
# # ax1.set_zlabel('Amplitude', fontproperties='Times New Roman', fontsize=13, rotation=90)
# x = np.linspace(-0.5, 5.3, 6)
# y = np.linspace(-0.5, 3.1, 4)
# X, Y = np.meshgrid(x, y)
# ax1.plot_surface(X,
#                 Y,
#                 Z=X*0+0.45,
#                 color='red',
#                 alpha=0.4
#                )
# x = np.linspace(-0.5, 5.3, 6)
# y = np.linspace(-0.5, 3.1, 4)
# X, Y = np.meshgrid(x, y)
# ax1.plot_surface(X,
#                 Y,
#                 Z=X*0+0.23,
#                 color='yellow',
#                 alpha=0.5
#                ) 
# ax1.text(0.6, -3.5, -0.4, '(a)', fontproperties='Arial', fontsize=12, fontdict={'weight':'bold'})
# plt.title('Traffic partterns of volume.                        ',
#           x=0.85,y=-0.24,
#           fontproperties='Times New Roman',
#           fontsize=12)

# # 第二个
# sr = 128  # 1.sampling rate
# wavename = 'morl'  # 2.母小波名称
# totalscal = 50  # 3.totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
# fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
# cparam = 2 * fc * totalscal  # 常数c
# scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
# [cwtmatr, frequencies] = pywt.cwt(sca_md[0, :], scales, wavename, 1.0 / sr)  # 4.y为将要进行cwt变换的一维输入信号
# t = np.arange(0, sca_md[0, :].shape[0]/sr, 1.0/sr)

# cwtmatr = np.flip(cwtmatr)

# ax2 = fig.add_subplot(132, projection='3d')
# print(t.shape)
# print(frequencies.shape)
# print(cwtmatr.shape)
# im2 = ax2.contour3D(t[1700:, ]/8-1.5, frequencies[:31, ]/14-1.7, abs(cwtmatr[:31, 1700:]), 150, cmap='rainbow')
# ax2.invert_xaxis()
# ax2.view_init(elev=15, azim=-45)
# ax2.tick_params(labelsize=10, pad=0.02)
# for l in ax2.xaxis.get_ticklabels():
#     l.set_family('Times New Roman')
# for l in ax2.yaxis.get_ticklabels():
#     l.set_family('Times New Roman')
# for l in ax2.zaxis.get_ticklabels():
#     l.set_family('Times New Roman')
# ax2.set_xlabel('Distance [km]', fontproperties='Times New Roman', fontsize=10, rotation=-20)
# ax2.set_ylabel('Time [hour]', fontproperties='Times New Roman', fontsize=10, rotation=60)
# # # ax2.set_zlabel('Amplitude', fontproperties='Times New Roman', fontsize=13, rotation=90)
# x = np.linspace(-0.5, 5.3, 6)
# y = np.linspace(-0.5, 3.1, 4)
# X, Y = np.meshgrid(x, y)
# ax2.plot_surface(X,
#                 Y,
#                 Z=X*0+0.5,
#                 color='red',
#                 alpha=0.4
#                )
# x = np.linspace(-0.5, 5.3, 6)
# y = np.linspace(-0.5, 3.1, 4)
# X, Y = np.meshgrid(x, y)
# ax2.plot_surface(X,
#                 Y,
#                 Z=X*0+0.3,
#                 color='yellow',
#                 alpha=0.5
#                ) 
# ax2.text(0.7, -3.3, -0.46, '(b)', fontproperties='Arial', fontsize=12, fontdict={'weight':'bold'})
# plt.title('Traffic partterns of density.                        ',
#           x=0.85,y=-0.24,
#           fontproperties='Times New Roman',
#           fontsize=12)

# 第三个
sr = 128  # 1.sampling rate
wavename = 'morl'  # 2.母小波名称
totalscal = 50  # 3.totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
cparam = 2 * fc * totalscal  # 常数c
scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
[cwtmatr, frequencies] = pywt.cwt(sca_sd[0, :], scales, wavename, 1.0 / sr)  # 4.y为将要进行cwt变换的一维输入信号
t = np.arange(0, sca_sd[0, :].shape[0]/sr, 1.0/sr)

# t = np.flip(t)
frequencies = np.flip(frequencies)
# cwtmatr = np.flip(cwtmatr)

ax3 = fig.add_subplot(133, projection='3d')
print(t.shape)
print(frequencies.shape)
print(cwtmatr.shape)
im3 = ax3.contour3D(t[3000:, ]/5-4.6, frequencies[25:45, ]/9-3.7, abs(cwtmatr[25:45, 3000:]), 150, cmap='rainbow')
ax3.invert_xaxis()
ax3.view_init(elev=15, azim=-45)
ax3.tick_params(labelsize=10, pad=0.02)
for l in ax3.xaxis.get_ticklabels():
    l.set_family('Times New Roman')
for l in ax3.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
for l in ax3.zaxis.get_ticklabels():
    l.set_family('Times New Roman')
ax3.set_xlabel('Distance [km]', fontproperties='Times New Roman', fontsize=10, rotation=-20)
ax3.set_ylabel('Time [hour]', fontproperties='Times New Roman', fontsize=10, rotation=60)
# ax3.set_zlabel('Amplitude', fontproperties='Times New Roman', fontsize=13, rotation=90)
x = np.linspace(-0.5, 5.3, 6)
y = np.linspace(-0.5, 3.1, 4)
X, Y = np.meshgrid(x, y)
ax3.plot_surface(X,
                Y,
                Z=X*0+0.3,
                color='red',
                alpha=0.4
               )
x = np.linspace(-0.5, 5.3, 6)
y = np.linspace(-0.5, 3.1, 4)
X, Y = np.meshgrid(x, y)
ax3.plot_surface(X,
                Y,
                Z=X*0+0.17,
                color='yellow',
                alpha=0.5
               ) 
# ax3.text(0.7, -3.2, -0.252, '(c)', fontproperties='Arial', fontsize=12, fontdict={'weight':'bold'})
# plt.title('Traffic partterns of speed.                          ',
#           x=0.86,y=-0.24,
#           fontproperties='Times New Roman',
#           fontsize=12)

plt.savefig(r'E:\2024huawei\article_figure\single_figure\交通速度模态.jpg',dpi=500,bbox_inches='tight')
plt.show()