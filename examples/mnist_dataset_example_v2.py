from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow_datasets as tfds
from cvnn import layers
from cvnn import losses
import numpy as np
import timeit
import datetime
from cvnn import metrics
from pdb import set_trace
try:
    import plotly.graph_objects as go
    import plotly
    PLOTLY = True
except ModuleNotFoundError:
    PLOTLY = False
import matplotlib.pyplot as plt 
from importlib import reload
import os
# tf.enable_v2_behavior()
# tfds.disable_progress_bar()


PLOTLY_CONFIG = {
    'scrollZoom': True,
    'editable': True
}

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def cast_to_complex(image, label):
    return tf.cast(image, tf.complex64), label


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    #ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def keras_fit(ds_train, ds_test, verbose=True, init1='glorot_uniform', init2='glorot_uniform', train_bias=True):
    tf.random.set_seed(24)
    # https://www.tensorflow.org/datasets/keras_example
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1), dtype=np.float32),
      tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init1, dtype=np.float32, use_bias=train_bias),
      tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=init2, dtype=np.float32, use_bias=train_bias)
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    weigths = model.get_weights()
    with tf.GradientTape() as tape:
        # for elem, label in iter(ds_train):
        elem, label = next(iter(ds_test))
        loss = model.compiled_loss(y_true=label, y_pred=model(elem))    # calculate loss
        gradients = tape.gradient(loss, model.trainable_weights)        # back-propagation
    logs = {
        'weights': weigths,
        'loss': loss,
        'gradients': gradients
    }
    start = timeit.default_timer()
    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    stop = timeit.default_timer()
    return history, stop - start, logs



def complex_model_only_dense():
    model = tf.keras.models.Sequential([
        layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.complex64),
        layers.ComplexDense(128, activation='cart_relu', dtype=np.complex64, kernel_initializer='glorot_uniform',
                            use_bias=False, init_technique='zero_imag'),
        layers.ComplexDense(10, activation='cast_to_real', dtype=np.complex64, kernel_initializer='glorot_uniform',
                            use_bias=False, init_technique='zero_imag'),
        tf.keras.layers.Activation('softmax')
    ])
    # model = tf.keras.models.Sequential([
    #     layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32),
    #     layers.ComplexDense(128, activation='cart_relu', dtype=np.float32, kernel_initializer=init1),
    #     layers.ComplexDense(10, activation='softmax_real_with_abs', dtype=np.float32, kernel_initializer=init2)
    # ])

    model.summary() 
    
    return model 

def complex_model_conv2d():
    tf.random.set_seed(1)
    init = 'ComplexGlorotUniform'
    acti = 'cart_relu'
    model = tf.keras.models.Sequential()
    model.add(layers.ComplexInput(input_shape=(28, 28, 1)))                     # Always use ComplexInput at the start
    model.add(layers.ComplexConv2D(32, (3, 3), activation=acti, input_shape=(32, 32, 3), kernel_initializer=init))
    model.add(layers.ComplexMaxPooling2D((2, 2)))
    model.add(layers.ComplexConv2D(64, (3, 3), activation=acti, kernel_initializer=init))
    model.add(layers.ComplexMaxPooling2D((2, 2)))
    model.add(layers.ComplexConv2D(64, (3, 3), activation=acti, kernel_initializer=init))
    model.add(layers.ComplexFlatten())
    model.add(layers.ComplexDense(64, activation=acti, kernel_initializer=init))
    model.add(layers.ComplexDense(10, activation='cart_softmax', kernel_initializer=init))
    
    model.summary()    
    return model 
    
    
def own_complex_fit_dense(ds_train, ds_test, verbose=True):
    tf.random.set_seed(24)
    
    model  = complex_model_only_dense() 
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    ds_train = ds_train.map(cast_to_complex)
    ds_test = ds_test.map(cast_to_complex)
    weigths = model.get_weights()
    with tf.GradientTape() as tape:
        # for elem, label in iter(ds_train):
        elem, label = next(iter(ds_test))
        loss = model.compiled_loss(y_true=label, y_pred=model(elem))  # calculate loss
        gradients = tape.gradient(loss, model.trainable_weights)  # back-propagation
    logs = {
        'weights': weigths,
        'loss': loss,
        'gradients': gradients
    }
    start = timeit.default_timer()
    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=verbose, shuffle=False
    )
    stop = timeit.default_timer()
    return history, stop - start, logs

def own_complex_fit_conv2d(epochs= 5):
    tf.random.set_seed(24)

    model = complex_model_conv2d()
    model.compile(optimizer='sgd', 
                      loss=losses.ComplexAverageCrossEntropy(),
                      metrics=metrics.ComplexCategoricalAccuracy())
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
         
    train_images, test_images = train_images.astype(dtype=np.complex64) / 255.0, test_images.astype(dtype=np.complex64) / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    tic = timeit.default_timer()
    history = model.fit(
        train_images, 
        train_labels, 
        epochs=epochs, 
        validation_data=(test_images, test_labels),
        batch_size=128,
        shuffle=False
    )
    toc = timeit.default_timer()
    
    print('total training time', (toc-tic))
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['complex_categorical_accuracy'])
    plt.plot(history.history['val_complex_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    


def test_mnist():
    assert not tf.test.gpu_device_name(), "Using GPU not good for debugging"
    ds_train, ds_test = get_dataset()
    # Don't use bias becase complex model gets a complex bias with imag not zero.
    keras_hist, keras_time, keras_logs = keras_fit(ds_train, ds_test, train_bias=False)
    keras_weigths = keras_logs['weights']
    own_cvnn_hist, own_cvnn_time, own_cvnn_logs = own_complex_fit(ds_train, ds_test)
    own_cvnn_weigths = own_cvnn_logs['weights']
    assert np.all([np.all(k_w == o_w) for k_w, o_w in zip(keras_weigths, own_cvnn_weigths[::2])])
    assert np.all([np.all(o_w == 0) for o_w in own_cvnn_weigths[1::2]])
    assert own_cvnn_logs['loss'] == keras_logs['loss']
    assert np.allclose(own_cvnn_logs['gradients'][2], keras_logs['gradients'][1])
    # for k, o in zip(keras_hist.history.values(), own_cvnn_hist.history.values()):
    #     assert np.allclose(k, o), f"\n{keras_hist.history}\n !=\n{own_cvnn_hist.history}"
    # DO AGAIN TO USE BIAS
    keras_hist, keras_time, keras_logs = keras_fit(ds_train, ds_test)
    keras_weigths = keras_logs['weights']
    own_hist, own_time, own_logs = own_complex_fit(ds_train, ds_test)
    own_weigths = own_logs['weights']
    assert [np.all(k_w == o_w) for k_w, o_w in zip(keras_weigths, own_weigths)]
    assert keras_hist.history == own_hist.history, f"\n{keras_hist.history}\n !=\n{own_hist.history}"
    assert own_logs['loss'] == keras_logs['loss']
    # for k, k2, o in zip(keras_hist.history.values(), keras2_hist.history.values(), own_hist.history.values()):
    #     if np.all(np.array(k) == np.array(k2)):
    #         assert np.all(np.array(k) == np.array(o)), f"\n{keras_hist.history}\n !=\n{own_hist.history}"
    

if __name__ == "__main__":
    
    epoch = 10 
  
    #test_mnist()
    # test_mnist_montecarlo()
    ds_train, ds_test = get_dataset()
    # keras_fit(ds_train, ds_test, train_bias=False)
    #own_complex_fit_dense(ds_train, ds_test)
    own_complex_fit_conv2d(epochs = epoch)


