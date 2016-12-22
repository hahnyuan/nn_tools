import tensorflow as tf

def session_config(gpu_mem_factor):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_factor
    return config