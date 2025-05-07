import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

import sys
sys.path.append("/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/")
from dopamine.adapt import tent
from batch_rl.multi_head.atari_helpers import QuantileLayerNormNetwork


def compute_entropy():
    test_online_convnet = QuantileLayerNormNetwork(num_actions=6, name="TTA")
    adapted_state_ph = tf.compat.v1.placeholder(
        observation_dtype, state_shape, name='adapted_state_ph')
    test_net_outputs = test_online_convnet(adapted_state_ph)
    test_q_entropy = tf.reduce_mean(tent.softmax_entropy(test_net_outputs.q_values), axis=0)


def main():
    # # 指定 TensorBoard 日志文件的路径
    # log_dir = '/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/logs_atari/Qbert/1%/CQL_ln_tta/eval_entropy0.1_visual/events.out.tfevents.1687696220.gpu016'

    # # 创建事件累加器，加载指定日志文件
    # ea = event_accumulator.EventAccumulator(log_dir)
    # ea.Reload()

    # # 打印所有的摘要事件
    # print(ea)
    # # for summary in ea.scalars.keys():
    # #     print(summary)


    # 创建一个 TensorFlow 图
    graph = tf.Graph()

    with graph.as_default():
        # 定义一个张量
        tensor = tf.constant([1, 2, 3], name='tensor')

        # 创建张量摘要
        tf.compat.v1.summary.tensor_summary('My Tensor Summary', tensor)

        # 将摘要信息写入 TensorBoard 日志文件
        with tf.compat.v1.summary.FileWriter('/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils', graph) as writer:
            # 创建一个会话并执行初始化操作
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())

                # 运行写入操作
                summary = sess.run(tf.compat.v1.summary.merge_all())
                writer.add_summary(summary)
    
    # 使用 EventAccumulator 加载日志文件
    ea = event_accumulator.EventAccumulator('/mnt/cephfs/home/lianzihao/code/transfer_rl/CQL/atari/batch_rl/utils')
    ea.Reload()

    # 打印所有的标签（summary keys）
    event_tags = ea.Tags()["tensors"]
    
    tensor_name = "My Tensor Summary"

    tag_data = ea.Tensors(tensor_name)
    print(tag_data)


    # 获取摘要信息
    # # summary = ea.SummaryByName('My Variable Summary')
    # summary = ea.Summary('My Variable Summary')

    # # 获取变量的值
    # variable_value_from_summary = summary.value[0].simple_value

    # print("Variable value from summary:", variable_value_from_summary)


if __name__ == "__main__":
    main()