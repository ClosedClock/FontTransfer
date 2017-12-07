import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.layers.python.layers import xavier_initializer, batch_norm

from DataLoader import TrainValSetLoader
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
batch_size = 20
load_size = 80
fine_size = 80
target_size = 40
original_font = 'Baoli'
target_font = 'Songti'

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 10000
do_training = True
do_validation = False
# do_comparison = True
on_server = True
step_display = 100
step_save = 1000
save_path = '../../saved_train_data/cnn_l1/style_transfer'
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
opt_data = {
    'data_root': '../../img/',
    'original_font': original_font,
    'target_font': target_font,
    'train_set_size': train_set_size,
    'val_set_size': val_set_size,
    'load_size': load_size,
    'fine_size': fine_size,
    'target_size': target_size,
    # 'original_mean': mean_map[original_font],
    # 'target_mean': mean_map[target_font],
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

        # if do_training:
        #     self.loader_train = DataLoaderDisk(**opt_data_train)
        # self.loader_val = DataLoaderDisk(**opt_data_val)

        self.loader = TrainValSetLoader(**opt_data)

        # self.session = tf.Session(graph=self.graph)
        # self.session.run(tf.global_variables_initializer())
        # print('finished')


    def build_graph(self):
        print('Building graph')

        with self.graph.as_default():
            # Input data
            self.images = tf.placeholder(tf.float32, shape=(None, fine_size, fine_size, 1))
            self.labels = tf.placeholder(tf.float32, shape=(None, target_size, target_size, 1))
            self.training = tf.placeholder(tf.bool)
            self.keep_dropout = tf.placeholder(tf.float32)

            global_step = tf.Variable(0,trainable=False)

            # 80 -> 40 -> 20
            conv1 = tf.layers.conv2d(self.images, filters=96, kernel_size=11, strides=2, padding='same',
                                     kernel_initializer = xavier_initializer(uniform=False))
            conv1 = batch_norm_layer(conv1, self.training, 'bn1')
            conv1 = tf.nn.relu(conv1)
            print('conv1 shape = %s' % conv1.shape)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding='same')
            print('pool1 shape = %s' % pool1.shape)

            # 20 -> 10
            conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=5, strides=1, padding='same',
                                     kernel_initializer = xavier_initializer(uniform=False))
            conv2 = batch_norm_layer(conv2, self.training, 'bn2')
            conv2 = tf.nn.relu(conv2)
            print('conv2 shape = %s' % conv2.shape)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=2, padding='same')
            print('pool2 shape = %s' % pool2.shape)

            # 10 -> 10
            conv3 = tf.layers.conv2d(pool2, filters=384, kernel_size=3, strides=1, padding='same',
                                     kernel_initializer = xavier_initializer(uniform=False))
            conv3 = batch_norm_layer(conv3, self.training, 'bn3')
            conv3 = tf.nn.relu(conv3)
            print('conv3 shape = %s' % conv3.shape)

            # 10 -> 10
            conv4 = tf.layers.conv2d(conv3, filters=256, kernel_size=3, strides=1, padding='same',
                                     kernel_initializer = xavier_initializer(uniform=False))
            conv4 = batch_norm_layer(conv4, self.training, 'bn4')
            conv4 = tf.nn.relu(conv4)
            print('conv4 shape = %s' % conv4.shape)


            fc5 = tf.reshape(conv4, [-1, 256 * 10 * 10])
            print('fc5 shape = %s' % fc5.shape)
            fc5 = tf.contrib.layers.fully_connected(fc5, 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc5 = batch_norm_layer(fc5, self.training, 'bn5')
            fc5 = tf.nn.dropout(fc5, self.keep_dropout)


            fc6 = tf.contrib.layers.fully_connected(fc5, 5000, tf.nn.relu,
                                                    weights_initializer=xavier_initializer(uniform=False))
            fc6 = batch_norm_layer(fc6, self.training, 'bn6')
            fc6 = tf.nn.dropout(fc6, self.keep_dropout)


            out = tf.contrib.layers.fully_connected(fc6, target_size * target_size, tf.nn.sigmoid,
                                                    weights_initializer=xavier_initializer(uniform=False))
            # out = tf.multiply(tf.add(tf.sign(tf.subtract(out, tf.constant(0.5))), tf.constant(1.)), 0.5)

            self.result = tf.reshape(out, [-1, target_size, target_size, 1])

            l1_loss = tf.losses.absolute_difference(self.labels, self.result)
            variation_loss = tf.image.total_variation(self.result)

            self.loss = tf.reduce_sum(l1_loss + variation_loss * 0)

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
                images_batch, labels_batch = self.loader.next_batch_train(batch_size)

                if step % step_display == 0:
                    print('[%s]:' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                    # Calculate batch loss and accuracy on training set
                    loss, result_batch = self.session.run([self.loss, self.result],
                                feed_dict={self.images: images_batch, self.labels: labels_batch,
                                           self.keep_dropout: 1., self.training: False})
                    print("-Iter " + str(step) + ", Training Loss= " + "{:.6f}".format(loss))
                    show_comparison(images_batch, labels_batch, result_batch,
                                    save=on_server, mode='display_train', iter=step)

                    # Calculate batch loss and accuracy on validation set
                    images_batch_val, labels_batch_val = self.loader.next_batch_val(batch_size)
                    loss, result_batch = self.session.run([self.loss, self.result],
                                feed_dict={self.images: images_batch_val, self.labels: labels_batch_val,
                                           self.keep_dropout: 1., self.training: False})
                    print("-Iter " + str(step) + ", Validation Loss= " + "{:.6f}".format(loss))
                    show_comparison(images_batch_val, labels_batch_val, result_batch,
                                    save=on_server, mode='display_val', iter=step)

                # Run optimization op (backprop)
                self.session.run(self.optimizer, feed_dict={self.images: images_batch, self.labels: labels_batch,
                                                            self.keep_dropout: dropout, self.training: True})
                # Save model
                if step != 0 and step % step_save == 0:
                    self.saver.save(self.session, save_path, global_step=step)
                    print('Model saved in file: %s at Iter-%d' % (save_path, step))

    def validate(self):
        print('Evaluation on the whole validation set...')

        num_batch = self.loader.size_val()//batch_size
        loss_total = 0.
        self.loader.reset_val()
        for i in range(num_batch):
            images_batch_val, labels_batch_val = self.loader.next_batch_val(batch_size)
            loss, result_batch = self.session.run([self.loss, self.result],
                            feed_dict={self.images: images_batch_val, self.labels: labels_batch_val,
                                       self.keep_dropout: 1., self.training: False})
            loss_total += loss
            print("Validation Loss = " + "{:.6f}".format(loss))
            show_comparison(images_batch_val, labels_batch_val, result_batch,
                            save=on_server, mode='validate', iter=i)

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