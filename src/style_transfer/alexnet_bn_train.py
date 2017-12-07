import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *


# Dataset Parameters
batch_size = 100
# batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
# training_iters = 50000
training_iters = 100000
do_training = False
do_validation = True
do_testing = False
step_display = 10
step_save = 5000
path_save = './alexnet_bn'
start_from = 'trained_model/one_more_layer/alexnet_bn-10000'
test_result_file = 'test_prediction.txt'

# # Start checking for rate reductions
# check_reduce_rate_threshold = 2000
# lowest_learning_rate = 0.000001
#
# # Iterations to check if average accuracy has increased
# check_reduce_rate = 1000 // step_display




def print_top_results(top_values, top_labels, top_values_relation, top_labels_relation, labels_batch):
    # assert tf.size(top_values) == tf.size(top_labels)
    k = top_values.shape[1]
    for i in range(batch_size):
        print('Top %d result for category (%s) -- After adding relation' % (k, words_list[labels_batch[i]]))
        for j in range(k):
            print('   %10.5f: %-20s %15.5f: %s' % (top_values[i, j], words_list[top_labels[i, j]],
                                                    top_values_relation[i, j], words_list[top_labels_relation[i, j]]))

def get_words_list():
    filename = '../../data/categories.txt'
    words_list = []
    with open(filename, 'r') as f:
        print('Files opens')
        for line in f.readlines():
            # print(line)
            word_with_number = line.split('/')[2]
            word = word_with_number.split(' ')[0]
            words_list.append(word.replace('_', ' '))
    return words_list

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc5-2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    #conv5 = conv4
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + Pool, 13->6
    conv5_2 = tf.nn.conv2d(conv5, weights['wc5-2'], strides=[1, 1, 1, 1], padding='SAME')
    conv5_2 = batch_norm_layer(conv5_2, train_phase, 'bn5-2')
    conv5_2 = tf.nn.relu(conv5_2)
    pool5_2 = tf.nn.max_pool(conv5_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5_2, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

opt_data_test = {
    #'data_h5': 'miniplaces_256_test.h5',
    'data_root': '../../data/images/test/',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

if do_training:
    loader_train = DataLoaderDisk(**opt_data_train)
if do_validation:
    loader_val = DataLoaderDisk(**opt_data_val)
if do_testing:
    loader_test = TestDataLoaderDisk(**opt_data_test)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# Matrix for relation, 1's for diagonal terms. Zero means for each column (diagonal not included)
relation_matrix = tf.cast(tf.constant(np.load('biased_relation.npz')['arr_0']), tf.float32)
relation_factor = 1.0 # How important possibilities of related categories are
words_list = get_words_list()

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = alexnet(x, keep_dropout, train_phase)
top5_values, top5_labels = tf.nn.top_k(logits, k=5)
# logits_relation = tf.matmul(logits, relation_matrix)
# top5_values_relation, top5_labels_relation = tf.nn.top_k(logits_relation, k=10)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
# accuracy1_relation = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_relation, y, 1), tf.float32))
# accuracy5_relation = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_relation, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
        print('Started from last time: %s' % start_from)
    else:
        sess.run(init)
        print('Initialized')

    if do_training:
        # # Previous top-5 accuracy
        # previous_acc5 = 0
        # # Vector of top-5 accuracies
        # acc5_vec = []

        for step in range(training_iters):
            # Load a batch of training data
            images_batch, labels_batch = loader_train.next_batch(batch_size)

            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                print("Learning rate is now " + str(learning_rate))

                # Calculate batch loss and accuracy on training set
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
                print("-Iter " + str(step) + ", Training Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5))

                # Calculate batch loss and accuracy on validation set
                images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False})
                print("-Iter " + str(step) + ", Validation Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5))

                # # Check if the accuracy is improving or not
                # if step >= check_reduce_rate_threshold % learning_rate > lowest_learning_rate:
                #     acc5_vec.append(acc5)
                #     if (step // step_display) % check_reduce_rate == 0:
                #         if sum(acc5_vec) / check_reduce_rate <= previous_acc5:
                #             learning_rate = learning_rate * 0.8
                #
                #         previous_acc5 = sum(acc5_vec) / check_reduce_rate
                #         acc5_vec = []

            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})

            # Save model
            if step % step_save == 0:
                saver.save(sess, path_save, global_step=step)
                print("Model saved at Iter %d !" %(step))

        print("Optimization Finished!")


    # Evaluate on the whole validation set
    if do_validation:
        print('Evaluation on the whole validation set...')
        num_batch = loader_val.size()//batch_size
        acc1_total = 0.
        acc5_total = 0.
        loader_val.reset()
        for i in range(num_batch//10):
            images_batch, labels_batch = loader_val.next_batch(batch_size)
            # t5_values, t5_labels, t5_values_relation, t5_labels_relation \
            #     = sess.run([top5_values, top5_labels, top5_values_relation, top5_labels_relation],
            #                                 feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            # print_top_results(t5_values, t5_labels, t5_values_relation, t5_labels_relation, labels_batch)
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            print("Validation Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

        acc1_total /= num_batch
        acc5_total /= num_batch
        print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))


    if do_testing:
        # Test on the test set
        print('Evaluation on the whole test set...')
        num_batch = loader_test.size()//batch_size
        loader_test.reset()

        with open(test_result_file, 'w') as f:
            print('Opened file %s' % test_result_file)
            for i in range(num_batch):
                print('There are %d test images left' % (loader_test.size() - i * batch_size))
                images_batch, filenames_batch = loader_test.next_batch(batch_size)
                # predicted_labels.shape = (batch_size, 5)
                predicted_labels = sess.run(top5_labels, feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})
                for j in range(len(filenames_batch)):
                    f.write(filenames_batch[j] + ' %d %d %d %d %d\n' % tuple(predicted_labels[j, :].tolist()))

        print('Test Finished!')

