import tensorflow as tf

# 构建模型
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(10)
])

# 构建输入样本和对应的目标值
input_data = tf.ones((1, 784))
target_value = tf.ones((1, 10))

# 计算梯度
with tf.GradientTape() as tape:
  # 前向传播
  predictions = model(input_data)
  # 计算损失函数
  loss_value = tf.keras.losses.MeanSquaredError()(target_value, predictions)

# 获取模型中所有可训练变量的梯度
gradients = tape.gradient(loss_value, model.trainable_variables)

# 打印梯度
for variable, gradient in zip(model.trainable_variables, gradients):
    print(variable.name, gradient)