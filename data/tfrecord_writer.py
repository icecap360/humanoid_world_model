import tensorflow as tf
import cv2
import os
import time
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def serialize_sample(img):
    image_shape = img.shape # tf.io.decode_jpeg(image_string).shape
    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
    #   'label': _int64_feature(label),
      'image_raw': _bytes_feature(tf.io.encode_png(img)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def encode_video(path_in, path_out):
    cap  =cv2.VideoCapture(path_in)
    assert cap.isOpened()
    with tf.io.TFRecordWriter(path_out) as writer:
        while(cap.isOpened()):
        # Capture each frame
            ret, frame = cap.read()
            if ret == True:
                writer.write(serialize_sample(frame))
            else:
                break
    cap.release()
    
def encode_dataset(mp4_videos_path, result_tfrecords_path):
    videos = os.listdir(mp4_videos_path)
    for video in videos:
        print('Working on Video')
        tfrecord = video.replace('.mp4', '.tfrecord')
        start = time.time()
        encode_video(os.path.join(mp4_videos_path, video), 
                      os.path.join(result_tfrecords_path, tfrecord))
        end = time.time()
        print('Took ', start - end, 'sec')

def load_dataset(dir):
    filenames = os.listdir(dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dir, filenames[i])
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        # 'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)
    
    raw_dataset = tf.data.TFRecordDataset(filenames).map(_parse_image_function)

if __name__ == '__main__':
    mp4_videos_path = '/pub0/qasim/1xgpt/data/data_v2_raw/train_v2.0_raw/videos'
    result_tfrecords_path = '/pub0/qasim/1xgpt/data/data_v2_raw/train_v2.0_raw/tfrecords/train'
    encode_dataset(mp4_videos_path, result_tfrecords_path)
    