"""
    Clear GPU memory utilities
    @note ref: https://www.kaggle.com/discussions/questions-and-answers/461668
"""
from numba import cuda
import tensorflow as tf
import gc

"""
    Pytorch
"""
device = cuda.get_current_device()
device.reset()

"""
    Tensorflow
"""
tf.keras.backend.clear_session()

"""
    General: Delete models and other variables using the GPU
"""
_ = gc.collect()