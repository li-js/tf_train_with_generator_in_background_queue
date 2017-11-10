#from __future__ import print_function

import sys,os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import setGPU
import tensorflow as tf
import os
import sys
import time

import numpy as np
from tqdm import tqdm 

BATCH_SIZE = 1
h, w = 321, 321

# generate a queue to pre-load the data into
image_batch0 = tf.placeholder(shape=(BATCH_SIZE,h,w,3), dtype=tf.float32)
label_batch0 = tf.placeholder(shape=(BATCH_SIZE,), dtype=tf.int32)

data_queue = tf.FIFOQueue(capacity=5, dtypes=[tf.float32, tf.int32], shapes=[(BATCH_SIZE,h,w,3), (BATCH_SIZE,)]) 
# data_queue accepts fixed shapes, can dequeu a batch of items with dequeue_op = data_queue.dequeue_many(BATCH)

data_queue2 = tf.FIFOQueue(capacity=5, dtypes=[tf.float32, tf.int32])  
# data_queue2 accepts variable shapes, but can only dequeu one item when shape varies. 

enqueue_op = data_queue.enqueue( (image_batch0, label_batch0) )
dequeue_op = data_queue.dequeue()
image_batch, label_batch = dequeue_op

# Define network and losses
net = DeepLabResNetModel({'data': image_batch}) # a dummy network
prediction = net['output']
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
reduced_loss = tf.reduce_mean(loss)
reduced_loss_with_l2 = reduced_loss + tf.add_n(l2_losses)

train_op = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(reduced_loss_with_l2)

# start a session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# a dummy data generator
def get_dummy_gen():  
  while True:
    Image_data, Image_lable = np.zeros((BATCH_SIZE,h,w,3)), np.ones((BATCH_SIZE,))
    yield Image_data, Image_lable

dummy_gen = get_dummy_gen()

coord = tf.train.Coordinator()
def enqueue_thread():
    with coord.stop_on_exception():
        while not coord.should_stop():
            Image_batch, Label_batch = dummy_gen.next()
            sess.run(enqueue_op, feed_dict={image_batch0: Image_batch, label_batch0: Label_batch})

import threading
numberOfThreads = 1
threads = []
for i in range(numberOfThreads): threads.append(threading.Thread(target=enqueue_thread))
for thread in threads: thread.start()

for step in tqdm(xrange(30)):
    #X, Y = sess.run(dequeue_op)
    #_, loss = sess.run([train_op, reduced_loss], feed_dict={image_batch:X, label_batch:Y})
    _, loss = sess.run([train_op, reduced_loss])


# Exit gracefully
coord.request_stop()
coord.join(threads)
    
