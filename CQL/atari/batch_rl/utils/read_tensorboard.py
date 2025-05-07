<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

point_colors = ['#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe', '#6c584c', '#e63946']

def draw_scatter(x, y, title, xlabel, ylabel, legend, path):
    # 不同的点显示不同的颜色
    colors = []
    for i in y:
        print(i)
        colors.append(point_colors[int(i)])

    # 绘制散点图
    plt.scatter(x, y, c=colors, s=0.1)

    # 设置标题、轴标签和图例
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(legend)

    # # 显示图形
    # plt.show()

    # 保存图片
    plt.savefig(path, dpi=300)

    # 清楚上一张图片
    plt.cla()


def draw_plot(x, y, title, xlabel, ylabel, legend, path):
    # 绘制散点图
    plt.plot(x, y)

    # 设置标题、轴标签和图例
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(legend)

    # # 显示图形
    # plt.show()

    # 保存图片
    plt.savefig(path, dpi=300)

    # 清楚上一张图片
    plt.cla()


def read():
    # 选择要读取的事件文件路径
    event_file = '/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta/eval_tent_visual/events.out.tfevents.1687750026.gpu016'

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取所有标签的列表
    tags = event_acc.Tags()['scalars']

    # 指定要读取的标签名称
    tag_name = 'test/reward'
    tta_tag_name = 'tta_1/action'

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)
    tta_tag_data = event_acc.Scalars(tta_tag_name)
    
    steps = []
    values = []
    tta_steps = []
    tta_values = []
    # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        # for scalar_event in tag_data[90000:]:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)

    for tta_scalar_event in tta_tag_data:
        # for tta_scalar_event in tta_tag_data[90000:]:
    # for scalar_event, tta_scalar_event in zip(tag_data, tta_tag_data):
        tta_step = tta_scalar_event.step
        tta_value = tta_scalar_event.value
        tta_steps.append(tta_step)
        tta_values.append(tta_value)
    
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/action distribution.png"
    tta_path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/tta action distribution.png"
    draw_entropy(steps, values, title="Action Distribution", xlabel="step", ylabel="action class", legend=["action plot"], path=path)
    draw_entropy(tta_steps, tta_values, title="TTA Action Distribution", xlabel="step", ylabel="action class", legend=["action plot"], path=tta_path)


    print(len(steps))
    print(len(tta_values))


def read_scalar(event_file, tag_name, title, ylabel, path, image_type):
    # 选择要读取的事件文件路径
    event_file = event_file

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)

    steps = []
    values = []
     # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)

    path = path

    if image_type == 'plot':
        draw_plot(steps, values, title=title, xlabel="step", ylabel=ylabel, legend=["Q value plot"], path=path)
    else:
        draw_scatter(steps, values, title=title, xlabel="step", ylabel=ylabel, legend=["Q value plot"], path=path)


def read_eval():
    # 选择要读取的事件文件路径
    event_file = '/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2/eval_entropy0.1_tent_visual/events.out.tfevents.1688898103.gpu016'

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取所有标签的列表
    tags = event_acc.Tags()['scalars']

    # 指定要读取的标签名称
    tag_name = 'eval/eval_q_value'
    tta_tag_name = 'tta_1/tta_q_value'

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)
    tta_tag_data = event_acc.Scalars(tta_tag_name)
    
    steps = []
    values = []
    tta_steps = []
    tta_values = []
    # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)
    
    for scalar_event in tta_tag_data:
        step = scalar_event.step
        value = scalar_event.value
        tta_steps.append(step)
        tta_values.append(value)

    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Q value.png"
    tta_path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/tta Q value.png"
    draw_plot(steps, values, title="normal", xlabel="step", ylabel="Q value", legend=["Q value plot"], path=path)
    draw_plot(tta_steps, tta_values, title="TTA", xlabel="step", ylabel="TTA Q value", legend=["Q value plot"], path=tta_path)


def main():
    # 选择要读取的事件文件路径
    event_file = '/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Pong/1%/CQL_ConvAndLn_1000it/eval_entropy0.1_tent/events.out.tfevents.1688907504.gpu016'
    tag_name = 'eval/eval_q_value'
    title = 'eval Q value'
    ylabel = "eval Q value"
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Pong eval Q value.png"
    read_scalar(event_file, tag_name, title, ylabel, path, image_type='plot')

    tag_name = 'tta_1/tta_q_value'
    title = 'tta Q value'
    ylabel = "tta Q value"
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Pong TTA Q value.png"
    read_scalar(event_file, tag_name, title, ylabel, path, image_type='plot')


    tag_name = 'eval/entorpy'
    title = 'eval entropy'
    ylabel = "eval entropy"
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Pong eval entropy.png"
    read_scalar(event_file, tag_name, title, ylabel, path, image_type='plot')

    tag_name = 'tta_1/entorpy'
    title = 'tta entropy'
    ylabel = "tta entropy"
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Pong TTA entropy.png"
    read_scalar(event_file, tag_name, title, ylabel, path, image_type='plot')


    tag_name = 'eval/action'
    title = 'eval action'
    ylabel = "eval action"
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Pong eval action.png"
    read_scalar(event_file, tag_name, title, ylabel, path, image_type='scatter')

    tag_name = 'tta_1/action'
    title = 'tta action'
    ylabel = "tta action"
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Pong TTA action.png"
    read_scalar(event_file, tag_name, title, ylabel, path, image_type='scatter')
    


if __name__ == "__main__":
=======
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

point_colors = ['#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe', '#6c584c', '#e63946']

def draw_multi_filter_plot(x1, y1, x2, y2, title, xlabel, ylabel, legend, path):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, marker='o', markersize=5, linewidth=1, label="action", color='blue')
    new_x2 = [x2[i] for i in range(len(y2)) if y2[i] != 0]
    new_y2 = [y for y in y2 if y != 0 ]
    print(x1)
    print(new_x2)
    ax.plot(new_x2, new_y2, marker='o', markersize=5, linewidth=1, label="reward", color='red')
    ax.set_xlabel(xlabel) #设置x轴名称 x label
    ax.set_ylabel(ylabel) #设置y轴名称 y label
    ax.set_title(title) #设置图名为Simple Plot
    ax.legend() #自动检测要在图例中显示的元素，并且显示

    # plt.show() #图形可视化

    # 保存图片
    plt.savefig(path, dpi=300)

def draw_multi_plot(x1, y1, x2, y2, title, xlabel, ylabel, legend, path):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label="baseline")
    ax.plot(x2, y2, label="TARL")
    ax.set_xlabel(xlabel) #设置x轴名称 x label
    ax.set_ylabel(ylabel) #设置y轴名称 y label
    ax.set_title(title) #设置图名为Simple Plot
    ax.legend() #自动检测要在图例中显示的元素，并且显示

    # plt.show() #图形可视化

    # 保存图片
    plt.savefig(path, dpi=300)

def read_data(event_file, tag_name):
    # 选择要读取的事件文件路径
    event_file = event_file

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)

    steps = []
    values = []
     # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)
    
    return steps, values

def draw_scatter(x, y, title, xlabel, ylabel, legend, path):
    # 不同的点显示不同的颜色
    colors = []
    for i in y:
        print(i)
        colors.append(point_colors[int(i)])

    # 绘制散点图
    plt.scatter(x, y, c=colors, s=0.1)

    # 设置标题、轴标签和图例
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(legend)

    # # 显示图形
    # plt.show()

    # 保存图片
    plt.savefig(path, dpi=300)

    # 清楚上一张图片
    plt.cla()


def draw_plot(x, y, title, xlabel, ylabel, legend, path):
    # 绘制散点图
    plt.plot(x, y)

    # 设置标题、轴标签和图例
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(legend)

    # # 显示图形
    # plt.show()

    # 保存图片
    plt.savefig(path, dpi=300)

    # 清楚上一张图片
    plt.cla()


def read():
    # 选择要读取的事件文件路径
    event_file = '/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta/eval_tent_visual/events.out.tfevents.1687750026.gpu016'

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取所有标签的列表
    tags = event_acc.Tags()['scalars']

    # 指定要读取的标签名称
    tag_name = 'test/reward'
    tta_tag_name = 'tta_1/action'

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)
    tta_tag_data = event_acc.Scalars(tta_tag_name)
    
    steps = []
    values = []
    tta_steps = []
    tta_values = []
    # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        # for scalar_event in tag_data[90000:]:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)

    for tta_scalar_event in tta_tag_data:
        # for tta_scalar_event in tta_tag_data[90000:]:
    # for scalar_event, tta_scalar_event in zip(tag_data, tta_tag_data):
        tta_step = tta_scalar_event.step
        tta_value = tta_scalar_event.value
        tta_steps.append(tta_step)
        tta_values.append(tta_value)
    
    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/action distribution.png"
    tta_path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/tta action distribution.png"
    draw_entropy(steps, values, title="Action Distribution", xlabel="step", ylabel="action class", legend=["action plot"], path=path)
    draw_entropy(tta_steps, tta_values, title="TTA Action Distribution", xlabel="step", ylabel="action class", legend=["action plot"], path=tta_path)


    print(len(steps))
    print(len(tta_values))


def read_scalar(event_file, tag_name, title, ylabel, path, image_type):
    # 选择要读取的事件文件路径
    event_file = event_file

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)

    steps = []
    values = []
     # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)

    path = path

    if image_type == 'plot':
        draw_plot(steps, values, title=title, xlabel="step", ylabel=ylabel, legend=["Q value plot"], path=path)
    else:
        draw_scatter(steps, values, title=title, xlabel="step", ylabel=ylabel, legend=["Q value plot"], path=path)


def read_eval():
    # 选择要读取的事件文件路径
    event_file = '/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ConvAnd_1000it_tta2/eval_entropy0.1_tent_visual/events.out.tfevents.1688898103.gpu016'

    # 创建EventAccumulator对象并加载事件文件
    event_acc = EventAccumulator(event_file)
    event_acc.Reload() 

    # 获取所有标签的列表
    tags = event_acc.Tags()['scalars']

    # 指定要读取的标签名称
    tag_name = 'eval/eval_q_value'
    tta_tag_name = 'tta_1/tta_q_value'

    # 获取指定标签的数据
    tag_data = event_acc.Scalars(tag_name)
    tta_tag_data = event_acc.Scalars(tta_tag_name)
    
    steps = []
    values = []
    tta_steps = []
    tta_values = []
    # 遍历输出每个步骤（step）和对应的值（value）
    for scalar_event in tag_data:
        step = scalar_event.step
        value = scalar_event.value
        steps.append(step)
        values.append(value)
    
    for scalar_event in tta_tag_data:
        step = scalar_event.step
        value = scalar_event.value
        tta_steps.append(step)
        tta_values.append(value)

    path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/Q value.png"
    tta_path = "/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils/figs/tta Q value.png"
    draw_plot(steps, values, title="normal", xlabel="step", ylabel="Q value", legend=["Q value plot"], path=path)
    draw_plot(tta_steps, tta_values, title="TTA", xlabel="step", ylabel="TTA Q value", legend=["Q value plot"], path=tta_path)


def main():
    # ####################################################################################################################################
    # # 2023/10/26 draw images
    # ####################################################################################################################################
    # file_prefix = "/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/"
    # # 选择要读取的事件文件路径
    # # event_file = file_prefix + 'logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct/eval_entropy0.1_kl1.5_augData/events.out.tfevents.1698309145.ts-ef879fe6b7d646d694902b991a384480-launcher'
    # event_file = "/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct/eval_entropy0.1_kl1.5_augData/events.out.tfevents.1698309145.ts-ef879fe6b7d646d694902b991a384480-launcher"
    
    # path = file_prefix + "atari/batch_rl/utils/figs/Asterix Q Value Estimates.png"
    # eval_q_steps, eval_q_values = read_data(event_file, "eval/eval_q_value")
    # tta_q_steps, tta_q_values = read_data(event_file, "tta_2/tta_q_value")
    # draw_multi_plot(eval_q_steps, eval_q_values, tta_q_steps, tta_q_values, "Asterix Q Value Estimates", "steps", "value estimates", "Q Value Estimates", path)

    # path = file_prefix + "atari/batch_rl/utils/figs/Asterix Entropy.png"
    # eval_q_steps, eval_q_values = read_data(event_file, "eval/entorpy")
    # tta_q_steps, tta_q_values = read_data(event_file, "tta_2/entorpy")
    # draw_multi_plot(eval_q_steps, eval_q_values, tta_q_steps, tta_q_values, "Asterix Entropy Comparison", "steps", "entropy", "Entropy Comparison", path)

    # path = file_prefix + "atari/batch_rl/utils/figs/Asterix Action.png"
    # eval_q_steps, eval_q_values = read_data(event_file, "eval/action")
    # tta_q_steps, tta_q_values = read_data(event_file, "tta_2/action")
    # draw_multi_plot(eval_q_steps, eval_q_values, tta_q_steps, tta_q_values, "Asterix Action Comparison", "steps", "action", "Action Comparison", path)

    # path = file_prefix + "atari/batch_rl/utils/figs/Asterix Reward.png"
    # eval_q_steps, eval_q_values = read_data(event_file, "test/reward")
    # tta_q_steps, tta_q_values = read_data(event_file, "tta/reward")
    # draw_multi_plot(eval_q_steps, eval_q_values, tta_q_steps, tta_q_values, "Asterix Reward Comparison", "steps", "reward", "Reward Comparison", path)
    # ####################################################################################################################################

    ####################################################################################################################################
    # 2023/10/27 draw images
    ####################################################################################################################################
    file_prefix = "/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/"
    event_file = "/apdcephfs/share_1594716/zihaolian/code/transfer_rl/CQL/logs_atari/Asterix/1%/CQL_ConvAndLn_1000it_correct/eval_entropy0.99_augData_record/events.out.tfevents.1698326659.ts-ef879fe6b7d646d694902b991a384480-launcher"
    path = file_prefix + "atari/batch_rl/utils/figs/Asterix Error Analysis.png"
    random_action_steps, random_action = read_data(event_file, "random/random_action")
    reward_steps, reward = read_data(event_file, "tta/reward")
    draw_multi_filter_plot(random_action_steps, random_action, reward_steps, reward, "Asterix Error Analysis", "steps", "value", "Error Analysis", path)
    ####################################################################################################################################

    # tag_name = 'eval/eval_q_value'
    # title = 'eval Q value'
    # ylabel = "eval Q value"
    # path = "atari/batch_rl/utils/figs/Pong eval Q value.png"
    # read_scalar(file_prefix + event_file, tag_name, title, ylabel, file_prefix + path, image_type='plot')

    # tag_name = 'tta_1/tta_q_value'
    # title = 'tta Q value'
    # ylabel = "tta Q value"
    # path = "atari/batch_rl/utils/figs/Pong TTA Q value.png"
    # read_scalar(file_prefix + event_file, tag_name, title, ylabel, file_prefix + path, image_type='plot')


    # tag_name = 'eval/entorpy'
    # title = 'eval entropy'
    # ylabel = "eval entropy"
    # path = "atari/batch_rl/utils/figs/Pong eval entropy.png"
    # read_scalar(file_prefix + event_file, tag_name, title, ylabel, file_prefix + path, image_type='plot')

    # tag_name = 'tta_1/entorpy'
    # title = 'tta entropy'
    # ylabel = "tta entropy"
    # path = "atari/batch_rl/utils/figs/Pong TTA entropy.png"
    # read_scalar(file_prefix + event_file, tag_name, title, ylabel, file_prefix + path, image_type='plot')


    # tag_name = 'eval/action'
    # title = 'eval action'
    # ylabel = "eval action"
    # path = "atari/batch_rl/utils/figs/Pong eval action.png"
    # read_scalar(file_prefix + event_file, tag_name, title, ylabel, file_prefix + path, image_type='scatter')

    # tag_name = 'tta_1/action'
    # title = 'tta action'
    # ylabel = "tta action"
    # path = "atari/batch_rl/utils/figs/Pong TTA action.png"
    # read_scalar(file_prefix + event_file, tag_name, title, ylabel, file_prefix + path, image_type='scatter')
    


if __name__ == "__main__":
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
    main()