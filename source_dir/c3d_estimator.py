import os

import tensorflow as tf

import source_dir.c3d_model as c3d_model


def _predict_result(logits):
    # Get the most common class from the list and the probability for this id must larger than 0.4
    # Otherwise return -1 as prediction result
    predicted_classes = tf.argmax(logits, 1)
    idx, _, count = tf.unique_with_counts(predicted_classes)
    count = tf.cast(count, tf.float32)
    count = count / tf.reduce_sum(count)
    class_ids = tf.cond(
        tf.greater(count[tf.argmax(count)], 0.4),
        lambda: idx[tf.argmax(count)], lambda: tf.constant(-1, tf.int64))

    predictions = {'class_ids': class_ids}
    export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs=export_outputs)


def model_fn(features, labels, mode, params):
    moving_average_decay = params['moving_average_decay']

    global_step = tf.train.get_or_create_global_step()

    # Define the Model
    logits = c3d_model.model(
        features['inputs'],
        height=params['height'],
        width=params['width'],
        channel=params['channel'],
        num_class=params['num_class'],
        dropout=0.5)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return _predict_result(logits)

    # Define the loss
    loss = c3d_model.loss(logits, labels)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        learning_rate=params['initial_learning_rate'],
        global_step=global_step,
        decay_steps=params['decay_step'],
        decay_rate=params['lr_decay_factor'],
        staircase=True)

    # Define the optimizer
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam')

    # Define the accuracy
    accuracy = c3d_model.accuracy(logits, labels)
    tf.summary.scalar('accuracy', accuracy[1])

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(train_op, variables_averages_op)

    # For evaluation
    eval_metric_ops = {'accuracy': accuracy}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def _decode_image(image, channel):
    image = tf.image.decode_jpeg(image, channels=channel)
    image = tf.cast(image, tf.float32)
    return image


def _decode_clip(clip, channel):
    image = tf.map_fn(lambda image: _decode_image(image, channel), clip)
    return image


def _parser(serialized_example, channel, num_frames_per_clip):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'clip/width': tf.FixedLenFeature([], tf.int64),
            'clip/height': tf.FixedLenFeature([], tf.int64),
            'clip/channel': tf.FixedLenFeature([], tf.int64),
            'clip/raw': tf.FixedLenFeature([num_frames_per_clip], tf.string),
            'clip/label': tf.FixedLenFeature([], tf.int64)
        })
    mapping_func = lambda image: _decode_image(image, channel)
    clip = tf.map_fn(mapping_func, features['clip/raw'], dtype=tf.float32)
    return clip, features['clip/label']


def train_input_fn(training_dir, params):
    dataset = tf.data.TFRecordDataset(
        os.path.join(training_dir, 'train.tfrecord'))
    dataset = dataset.shuffle(buffer_size=params['train_total_video_clip'])
    dataset = dataset.map(
        map_func=
        lambda serialized_example: _parser(serialized_example, params['channel'], params['num_frames_per_clip'])
    )
    dataset = dataset.repeat()
    iterator = dataset.batch(
        batch_size=params['batch_size']).make_one_shot_iterator()
    clips, labels = iterator.get_next()
    return {'inputs': clips}, labels


def eval_input_fn(training_dir, params):
    dataset = tf.data.TFRecordDataset(
        os.path.join(training_dir, 'eval.tfrecord'))
    dataset = dataset.shuffle(buffer_size=params['eval_total_video_clip'])
    dataset = dataset.map(
        map_func=
        lambda serialized_example: _parser(serialized_example, params['channel'], params['num_frames_per_clip'])
    )
    iterator = dataset.batch(
        batch_size=params['batch_size']).make_one_shot_iterator()
    clips, labels = iterator.get_next()
    return {'inputs': clips}, labels


def serving_input_fn(params):
    inputs = {
        'inputs':
        tf.placeholder(
            tf.float32,
            shape=[
                None, params['num_frames_per_clip'], params['height'],
                params['width'], params['channel']
            ])
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)()
