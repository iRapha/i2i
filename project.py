import os
import numpy as np
import tensorflow as tf

from shutil import copyfile
from tensorflow.contrib.tensorboard.plugins import projector
from utils import ensure_dir_exists

dataset = 'handbags_val'
LOG_DIR = 'summaries/projector/{}'.format(dataset)
ensure_dir_exists(LOG_DIR)

with open('summaries/color/{}.csv'.format(dataset), 'r') as f:
    csv = f.read().split('\n')
    N = len(csv) - 1 # Number of items.
    D = 3 # Dimensionality of the embedding.
    all_ex = np.zeros([N, D])
    for i, ex in enumerate(csv):
        if ex == '': continue
        ex_split = ex.split(', ')
        all_ex[i, :] = [int(x) for x in ex_split[0:3]]

embedding_var = tf.Variable(tf.constant(all_ex))

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = 'meta.tsv'
# our tsv is in summaries/color/{}.tsv, we want it at summaries/projector/{}/meta.tsv
copyfile('summaries/color/{}.tsv'.format(dataset), os.path.join(LOG_DIR, 'meta.tsv'))

embedding.sprite.image_path = 'sprite.png'.format(dataset)
# our sprite is in summaries/color/{}.png, we want it at summaries/projector/{}/sprite.png
copyfile('summaries/color/{}.png'.format(dataset), os.path.join(LOG_DIR, 'sprite.png'))
# Specify the width and height of a single thumbnail.
embedding.sprite.single_image_dim.extend([5, 5])

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)
