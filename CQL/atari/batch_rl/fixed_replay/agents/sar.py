<<<<<<< HEAD
# the tensorflow version of Sar

import tensorflow as tf

def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers:
        layer.trainable = False
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            # force use of batch stats in train and eval modes
            layer.track_running_stats = False
            layer.moving_mean = None
            layer.moving_variance = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.GroupNormalization)):
            layer.trainable = True
=======
# the tensorflow version of Sar

import tensorflow as tf

def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers:
        layer.trainable = False
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            # force use of batch stats in train and eval modes
            layer.track_running_stats = False
            layer.moving_mean = None
            layer.moving_variance = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.GroupNormalization)):
            layer.trainable = True
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
    return model