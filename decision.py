plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6, 4))
plot_decision_regions(X=x_tsne, y=y_test, classifier=svc)
plt.xlabel('Dimension 1', fontproperties='Times New Roman', fontsize=12)
plt.ylabel('Dimension 2', fontproperties='Times New Roman', fontsize=12)
# plt.title('Visualization of KGM decision results', fontproperties='Times New Roman', fontsize=15)
plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size':12})
plt.xticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
plt.yticks(fontproperties='Times New Roman', fontsize=11, rotation=0)
plt.tight_layout()
plt.grid(None)
bwith = 1 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['top'].set_color('black')  # 设置上‘脊梁’为红色
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
scale_x = range(-6,10,3)
index_x = ['-0.8', '-0.4', '0', '0.4', '0.8', '1.2']
plt.xticks(scale_x, index_x)
scale_y = range(-8,7,2)
index_y = ['-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4', '0.6']
plt.yticks(scale_y, index_y)
# 图例空白背景
rect5 = plt.Rectangle((4.5,4.9),4.2,2.4, 
                        fill=True,
                        color="white",
                       linewidth=0, zorder=5)
                       #facecolor="red")
plt.gca().add_patch(rect5)
# 图例黑色方框
rect6 = plt.Rectangle((4.5,4.9),4.2,2.4, 
                        fill=False,
                        color="black",
                       linewidth=1, zorder=5)
                       #facecolor="red")
plt.gca().add_patch(rect6)
plt.show()