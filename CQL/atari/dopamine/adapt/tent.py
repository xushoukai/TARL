<<<<<<< HEAD
# the tensorflow version of Tent

import copy
import tensorflow as tf


class Tent(tf.keras.Model):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @tf.function
    def call(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@tf.function
def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -tf.reduce_sum(tf.nn.softmax(x, axis=-1) * 
                          tf.math.log(tf.nn.softmax(x, axis=-1)), axis=-1)


@tf.function
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    with tf.GradientTape() as tape:
        # forward
        outputs = model(x)
        # adapt
        loss = tf.reduce_mean(softmax_entropy(outputs), axis=0)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's layers and collect all normalization layer parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.BatchNormalization, 
                              tf.keras.layers.LayerNormalization, 
                              tf.keras.layers.GroupNormalization)):
            for w in layer.weights:
                if 'gamma' in w.name:
                    params.append(w)
                    names.append(f"{layer.name}.{'weight'}")
                elif 'beta' in w.name:
                    params.append(w)
                    names.append(f"{layer.name}.{'bias'}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.get_weights())
    optimizer_state = copy.deepcopy(optimizer.get_weights())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.set_weights(model_state)
    optimizer.set_weights(optimizer_state)


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
        # if isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.GroupNormalization)):
        if isinstance(layer, (tf.keras.layers.LayerNormalization)):
            layer.trainable = True
    return model


def configure_final_ln_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers:
        layer.trainable = False
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    layers = []
    for layer in model.layers:
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        # if isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.GroupNormalization)):
        if isinstance(layer, (tf.keras.layers.LayerNormalization)):
            layer.trainable = False
            layers.append(layer)
    
    (layers[-1]).trainable = True

    return model


def finetune_final_layer(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers[:-1]:
        layer.trainable = False
    final_layer = model.layers[-1]
    final_layer.trainable = True
    return model


def configure_final_layer(model):
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
    return model


def configure_final_ln_layer(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers:
        layer.trainable = False
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    layer = model.layers[-2]
    layer.trainable = True
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = tf.keras.backend.learning_phase()
    assert is_training, "tent needs train mode: set_learning_phase(True)"
    param_grads = [p.trainable for p in model.trainable_variables]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which are trainable"
    assert not has_all_params, "tent should not update all params: " \
                               "check which are trainable"
    has_bn = any([isinstance(layer, (tf.keras.layers.BatchNormalization, 
                                     tf.keras.layers.LayerNormalization,
                                     tf.keras.layers.GroupNormalization)) 
                  for layer in model.layers])
=======
# the tensorflow version of Tent

import copy
import tensorflow as tf


class Tent(tf.keras.Model):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @tf.function
    def call(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@tf.function
def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -tf.reduce_sum(tf.nn.softmax(x, axis=-1) * 
                          tf.math.log(tf.nn.softmax(x, axis=-1)), axis=-1)


@tf.function
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    with tf.GradientTape() as tape:
        # forward
        outputs = model(x)
        # adapt
        loss = tf.reduce_mean(softmax_entropy(outputs), axis=0)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's layers and collect all normalization layer parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.BatchNormalization, 
                              tf.keras.layers.LayerNormalization, 
                              tf.keras.layers.GroupNormalization)):
            for w in layer.weights:
                if 'gamma' in w.name:
                    params.append(w)
                    names.append(f"{layer.name}.{'weight'}")
                elif 'beta' in w.name:
                    params.append(w)
                    names.append(f"{layer.name}.{'bias'}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.get_weights())
    optimizer_state = copy.deepcopy(optimizer.get_weights())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.set_weights(model_state)
    optimizer.set_weights(optimizer_state)


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
        # if isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.GroupNormalization)):
        if isinstance(layer, (tf.keras.layers.LayerNormalization)):
            layer.trainable = True
    return model


def configure_final_ln_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers:
        layer.trainable = False
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    layers = []
    for layer in model.layers:
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        # if isinstance(layer, (tf.keras.layers.LayerNormalization, tf.keras.layers.GroupNormalization)):
        if isinstance(layer, (tf.keras.layers.LayerNormalization)):
            layer.trainable = False
            layers.append(layer)
    
    (layers[-1]).trainable = True

    return model


def finetune_final_layer(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers[:-1]:
        layer.trainable = False
    final_layer = model.layers[-1]
    final_layer.trainable = True
    return model


def configure_final_layer(model):
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
    return model


def configure_final_ln_layer(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    tf.keras.backend.set_learning_phase(True)
    # disable grad, to (re-)enable only what SAR updates
    for layer in model.layers:
        layer.trainable = False
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    layer = model.layers[-2]
    layer.trainable = True
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = tf.keras.backend.learning_phase()
    assert is_training, "tent needs train mode: set_learning_phase(True)"
    param_grads = [p.trainable for p in model.trainable_variables]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which are trainable"
    assert not has_all_params, "tent should not update all params: " \
                               "check which are trainable"
    has_bn = any([isinstance(layer, (tf.keras.layers.BatchNormalization, 
                                     tf.keras.layers.LayerNormalization,
                                     tf.keras.layers.GroupNormalization)) 
                  for layer in model.layers])
>>>>>>> ed0d7b9c1ae446c6cffed41684f9178952a22685
    assert has_bn, "tent needs normalization for its optimization"