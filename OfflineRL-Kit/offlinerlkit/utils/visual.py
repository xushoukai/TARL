import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
 
#加载日志数据
# ea=event_accumulator.EventAccumulator('/apdcephfs/share_1594716/zihaolian/code/transfer_rl/log_OfflineRL-kit/walker2d-medium-v2/eval_mcql_ln_vae_trainval0.8/seed_0&timestamp_23-1007-193411/record/tb/events.out.tfevents.1696678451.ts-cb59d9b9ba9c45908af353035cf46210-launcher.486.0') 
ea=event_accumulator.EventAccumulator('/apdcephfs/share_1594716/zihaolian/code/transfer_rl/log_OfflineRL-kit/walker2d-medium-v2/eval_mcql_ln_vae_trainval0.99/seed_0&timestamp_23-1008-174256/record/tb/events.out.tfevents.1696758177.ts-5215df98d1c04367b4ea9e44e2d862a2-launcher.304.0') 
ea.Reload()
print(ea.scalars.Keys())
 
offline_data=ea.scalars.Items('offline/vae_loss')
online_data=ea.scalars.Items('online/vae_loss')
tta_data=ea.scalars.Items('tta/vae_loss')
offline_list = [i.value for i in offline_data]
online_list = [i.value for i in online_data]
tta_list = [i.value for i in tta_data]
# print([(i.step,i.value) for i in offline_lists])

def drop_histogram(data_list, bins, title, color="blue", alpha=0.5, label='Histogram'):
    # 设置最大的显示范围
    xmin, xmax = 0, 0.3
    # 使用numpy的histogram函数将数据分成100个区间，并统计每个区间内数据的数量
    hist, bins = np.histogram(data_list, bins=bins)
    # 计算频率
    freq = hist / (len(data_list) * np.diff(bins) * 100)
    # 使用matplotlib的bar函数将hist和bins绘制成直方图
    plt.bar(bins[:-1], freq, width=np.diff(bins), align='edge', color=color, alpha=alpha, label=label)
    # 添加图例
    plt.legend()
    # plt.show()
    plt.xlim(xmin, xmax)
    # plt.ylim(0, freq.max() * 1.1)

    plt.savefig(title)

    # foo_fig = plt.gcf() # 'get current figure'
    # foo_fig.savefig(title, format='eps', dpi=1000)
    # plt.show()

def main():
    bins = 100
    # offline_title = "offline.eps"
    offline_title = "offline_0.99.png"
    # offline_title = "offline.png"
    # offline_color = "#adb5bd"
    offline_color = "#cdb4db"
    offline_alpha = 1.0
    offline_label = "offline"
    # online_title = "online.eps"
    # online_title = "online.png"
    online_title = "online_0.99.png"
    online_color = "#bde0fe"
    online_alpha = 0.8
    online_label = "online"
    # tta_title = "tta.eps"
    # tta_title = "tta.png"
    tta_title = "tta_0.99.png"
    tta_color = "#ffafcc"
    tta_alpha = 0.6
    tta_label = "TARL"
    
    drop_histogram(offline_list, bins, offline_title, color=offline_color, alpha=offline_alpha, label=offline_label)
    drop_histogram(online_list, bins, online_title, color=online_color, alpha=online_alpha, label=online_label)
    drop_histogram(tta_list, bins, tta_title, color=tta_color, alpha=tta_alpha, label=tta_label)


if __name__=="__main__":
    main()