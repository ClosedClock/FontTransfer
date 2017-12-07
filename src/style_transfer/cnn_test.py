import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.layers.python.layers import xavier_initializer, batch_norm

from DataLoader import *
from display import show_comparison

POOL1 = False  # Set to True to add pooling after first conv layer
POOL2 = False  # Set to True to add pooling after second conv layer
POOL3 = True  # Set to True to add pooling after third conv layer
BN = False  # Set to True to use batch normalization
BIAS = True

train_set_size = 3000
val_set_size = 450
assert train_set_size + val_set_size <= 3498

# Dataset Parameters
batch_size = 100
load_size = 30
fine_size = 30
original_font = 'Baoli'
target_font = 'Songti'

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 10000
do_training = True
do_validation = False
do_comparison = True
step_display = 100
step_save = 1000
save_path = '../../saved_train_data/sigmoid/style_transfer'
# start_from = save_path + '-0'
start_from = ''

# mean values of images for each font
mean_map = {
    'Cao': 0.8508965474736796,
    'Hannotate': 0.74494465944527832,
    'Xingkai': 0.76178657894442514,
    'simkai': 0.8331927743152947,
    'STHeiti': 0.68690325752875914,
    'Songti': 0.63991741282015269,
    'Hanzipen': 0.79626192713369814,
    'Baoli': 0.76018856022486725,
    'WeibeiSC': 0.78538279635254127,
    'Yuanti': 0.5077452970321058
}

# Construct dataloader
opt_data_train = {
    'data_root': '../../img/',
    'original_font': original_font,
    'target_font': target_font,
    'start_index': 0,
    'number': train_set_size,
    'load_size': load_size,
    'fine_size': fine_size,
    'original_mean': mean_map[original_font],
    'target_mean': mean_map[target_font],
    'randomize': False
}

opt_data_val = {
    'data_root': '../../img/',
    'original_font': original_font,
    'target_font': target_font,
    'start_index': train_set_size,
    'number': val_set_size,
    'load_size': load_size,
    'fine_size': fine_size,
    'original_mean': mean_map[original_font],
    'target_mean': mean_map[target_font],
    'randomize': False
}

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)


class CharacterTransform:
    def __init__(self):
        '''Initialize the class by loading the required datasets 
        and building the graph'''
        self.graph = tf.Graph()
        self.build_graph()

        if do_training:
            self.loader_train = DataLoaderDisk(**opt_data_train)
        self.loader_val = DataLoaderDisk(**opt_data_val)

        # self.session = tf.Session(graph=self.graph)
        # self.session.run(tf.global_variables_initializer())
        # print('finished')


    def build_graph(self):
        print('Building graph')

        with self.graph.as_default():
            # Input data
            self.images = tf.placeholder(tf.float32, shape=(None, fine_size, fine_size))
            self.labels = tf.placeholder(tf.float32, shape=(None, fine_size, fine_size))
            self.training = tf.placeholder(tf.bool)
            self.keep_dropout = tf.placeholder(tf.float32)

            global_step = tf.Variable(0,trainable=False)

            fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.images, [-1, fine_size * fine_size]), 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc1 = batch_norm_layer(fc1, self.training, 'bn1')
            fc1 = tf.nn.dropout(fc1, self.keep_dropout)


            fc2 = tf.contrib.layers.fully_connected(fc1, 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc2 = batch_norm_layer(fc2, self.training, 'bn2')
            fc2 = tf.nn.dropout(fc2, self.keep_dropout)

            fc3 = tf.contrib.layers.fully_connected(fc2, 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc3 = batch_norm_layer(fc3, self.training, 'bn3')
            fc3 = tf.nn.dropout(fc3, self.keep_dropout)

            fc4 = tf.contrib.layers.fully_connected(fc3, 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc4 = batch_norm_layer(fc4, self.training, 'bn4')
            fc4 = tf.nn.dropout(fc4, self.keep_dropout)

            fc5 = tf.contrib.layers.fully_connected(fc4, 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc5 = batch_norm_layer(fc5, self.training, 'bn5')
            fc5 = tf.nn.dropout(fc5, self.keep_dropout)

            out = tf.contrib.layers.fully_connected(fc5, fine_size * fine_size, tf.nn.sigmoid,
                                                    weights_initializer=xavier_initializer(uniform=False))
            # out = tf.multiply(tf.add(tf.sign(tf.subtract(out, tf.constant(0.5))), tf.constant(1.)), 0.5)

            self.result = tf.reshape(out, [-1, fine_size, fine_size])  # trying to define a variable to store results

            self.loss = tf.reduce_sum(tf.losses.mean_squared_error(out, tf.reshape(self.labels, [-1, fine_size * fine_size])))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver() #### define the saving part

        print('Graph building finished')

    def train_model(self):
        '''Train the model with minibatches in a tensorflow session'''

        with tf.Session(graph=self.graph) as self.session:

            if len(start_from) > 1:
                self.saver.restore(self.session, start_from)
                print('Started from last time: %s' % start_from)
            else:
                self.session.run(tf.global_variables_initializer())
                print('Initialized')

            for step in range(training_iters):
                # Data to feed into the placeholder variables in the tensorflow graph
                images_batch, labels_batch = self.loader_train.next_batch(batch_size)

                if step % step_display == 0:
                    print('[%s]:' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                    # Calculate batch loss and accuracy on training set
                    loss, result_batch = \
                        self.session.run([self.loss, self.result], feed_dict={self.images: images_batch, self.labels: labels_batch,
                                                                              self.keep_dropout: 1., self.training: False})
                    print("-Iter " + str(step) + ", Training Loss= " + "{:.6f}".format(loss))
                    if do_comparison:
                        show_comparison(images_batch, labels_batch, result_batch)

                    # Calculate batch loss and accuracy on validation set
                    images_batch_val, labels_batch_val = self.loader_val.next_batch(batch_size)
                    loss, result_batch =\
                        self.session.run([self.loss, self.result], feed_dict={self.images: images_batch_val, self.labels: labels_batch_val,
                                                                              self.keep_dropout: 1., self.training: False})
                    print("-Iter " + str(step) + ", Validation Loss= " + "{:.6f}".format(loss))
                    if do_comparison:
                        show_comparison(images_batch_val, labels_batch_val, result_batch)

                # Run optimization op (backprop)
                self.session.run(self.optimizer, feed_dict={self.images: images_batch, self.labels: labels_batch,
                                                            self.keep_dropout: dropout, self.training: True})
                # Save model
                if step != 0 and step % step_save == 0:
                    self.saver.save(self.session, save_path, global_step=step)
                    print('Model saved in file: %s at Iter-%d' % (save_path, step))

    def validate(self):
        print('Evaluation on the whole validation set...')

        num_batch = self.loader_val.size()//batch_size
        loss_total = 0.
        self.loader_val.reset()
        for i in range(num_batch):
            images_batch_val, labels_batch_val = self.loader_val.next_batch(batch_size)
            loss, result_batch = self.session.run([self.loss, self.result], feed_dict={self.images: images_batch_val, self.labels: labels_batch_val,
                                                     self.keep_dropout: 1., self.training: False})
            loss_total += loss
            print("Validation Loss = " + "{:.6f}".format(loss))
            if do_comparison:
                show_comparison(images_batch_val, labels_batch_val, result_batch)

        loss_total /= num_batch
        print("Evaluation Finished! Loss = " + "{:.6f}".format(loss_total))

    def prediction(self, image_batch):
        pass

    def run(self):
        if do_training:
            self.train_model()
        if do_validation:
            self.validate()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.session.close()


if __name__ == '__main__':
    with CharacterTransform() as conv_net:
        conv_net.run()