import tensorflow as tf

import source_dir.c3d_estimator as c3d_estimator

MODEL_DIR = 'c3d_result'
DATA_DIR = '/Users/dongfenggu/Desktop/tfrecord'

HYPERPARAMETERS = {
    'num_class': 6,
    'batch_size': 20,
    'width': 120,
    'height': 100,
    'channel': 3,
    'num_frames_per_clip': 16,
    'train_total_video_clip': 300000,
    'eval_total_video_clip': 100000,
    'moving_average_decay': 0.9999,
    'initial_learning_rate': 1e-4,
    'decay_step': 10000,
    'lr_decay_factor': 0.1
}

config = tf.estimator.RunConfig(
    log_step_count_steps=1, save_summary_steps=1, model_dir=MODEL_DIR)

classifier = tf.estimator.Estimator(
    model_fn=c3d_estimator.model_fn, params=HYPERPARAMETERS, config=config)

classifier.train(
    input_fn=lambda: c3d_estimator.train_input_fn(DATA_DIR, HYPERPARAMETERS),
    steps=100)

classifier.evaluate(
    input_fn=lambda: c3d_estimator.eval_input_fn(DATA_DIR, HYPERPARAMETERS),
    steps=10)

classifier.export_saved_model(
    export_dir_base=MODEL_DIR,
    serving_input_receiver_fn=
    lambda: c3d_estimator.serving_input_fn(HYPERPARAMETERS))
