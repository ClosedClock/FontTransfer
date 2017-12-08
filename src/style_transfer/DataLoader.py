import os
import numpy as np
import scipy.misc
import datetime
import glob
np.random.seed(123)


class TrainValSetLoader(object):
    def __init__(self, **kwargs):
        """
        Initialize this data loader with given parameters
        :param kwargs: parameters
        """
        load_size = int(kwargs['load_size'])
        fine_size = int(kwargs['fine_size'])
        target_size = int(kwargs['target_size'])
        # self.original_mean = np.array(kwargs['original_mean'])
        # self.target_mean = np.array(kwargs['target_mean'])
        # self.randomize = kwargs['randomize']
        original_dir = os.path.join(kwargs['data_root'], kwargs['original_font'])
        target_dir = os.path.join(kwargs['data_root'], kwargs['target_font'])

        train_set_size = kwargs['train_set_size']
        val_set_size = kwargs['val_set_size']

        # Get lists of all images with original font and target font
        list_original = sorted(glob.glob(os.path.join(original_dir, '*.png')))
        list_target = sorted(glob.glob(os.path.join(target_dir, '*.png')))
        assert len(list_original) == len(list_target)
        total_size = len(list_original)

        list_original = np.array(list_original, np.object)
        list_target = np.array(list_target, np.object)

        # Permutation. Can be fixed by feeding a seed
        if kwargs['user'] == 'zijinshi':
            np.random.seed(42)
        else:
            np.random.seed(42)
        perm = np.random.permutation(total_size)
        list_original[:] = list_original[perm]
        list_target[:] = list_target[perm]

        print('Train set size = %d' % train_set_size)
        print('Validation set size = %d' % val_set_size)

        # Construct two sub dataloaders
        train_original = list_original[0:train_set_size]
        train_target   = list_target  [0:train_set_size]
        self.train_set = DataLoaderDisk(train_original, train_target, load_size, fine_size, target_size)
        val_original   = list_original[train_set_size:(train_set_size + val_set_size)]
        val_target     = list_target  [train_set_size:(train_set_size + val_set_size)]
        self.val_set = DataLoaderDisk(val_original, val_target, load_size, fine_size, target_size)

    def next_batch_train(self, batch_size):
        """
        Return next batch of data on training set
        :param batch_size: batch size
        :return: tuple, (original_batch, target_batch)
        """
        return self.train_set.next_batch(batch_size)

    def next_batch_val(self, batch_size):
        """
        Return next batch of data on validation set
        :param batch_size: batch size
        :return: tuple, (original_batch, target_batch)
        """
        return self.val_set.next_batch(batch_size)

    def size_train(self):
        return self.train_set.size()

    def size_val(self):
        return self.val_set.size()

    def reset_val(self):
        """Reset the inner counter of validation set to 0 (start from the beginning). """
        self.val_set.reset()


# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, list_original, list_target, load_size, fine_size, target_size):

        self.load_size = load_size
        self.fine_size = fine_size
        self.target_size = target_size
        # self.original_mean = np.array(kwargs['original_mean'])
        # self.target_mean = np.array(kwargs['target_mean'])
        # self.randomize = kwargs['randomize']

        self.list_original = list_original
        self.list_target = list_target

        assert len(list_original) == len(list_target)
        self.num = len(list_original)

        self._idx = 0
        
    def next_batch(self, batch_size):
        original_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 1))
        target_batch = np.zeros((batch_size, self.target_size, self.target_size, 1))
        for i in range(batch_size):
            original_image = scipy.misc.imread(self.list_original[self._idx])
            original_image = scipy.misc.imresize(original_image, (self.load_size, self.load_size))
            original_image = original_image.astype(np.float32)/255.
            # Revert to white characters on black background
            # original_image = -(original_image - self.original_mean)
            original_image = 1 - original_image

            target_image = scipy.misc.imread(self.list_target[self._idx])
            target_image = scipy.misc.imresize(target_image, (self.target_size, self.target_size))
            target_image = target_image.astype(np.float32)/255.
            # target_image = -(target_image - self.target_mean)
            target_image = 1 - target_image
            # if self.randomize:
            #     flip = np.random.random_integers(0, 1)
            #     if flip>0:
            #         original_image = original_image[:, ::-1]
            #     offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
            #     offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            # else:
            #     offset_h = (self.load_size-self.fine_size)//2
            #     offset_w = (self.load_size-self.fine_size)//2
            # original_batch[i, :, :, 0] = \
            #     original_image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]
            # target_batch[i, :, :] \
            #     = target_image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]

            original_batch[i, :, :, 0] = original_image
            target_batch[i, :, :, 0] = target_image

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return original_batch, target_batch

    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
