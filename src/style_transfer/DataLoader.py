import os
import numpy as np
import scipy.misc
import datetime
import glob
np.random.seed(123)

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.target_size = int(kwargs['target_size'])
        self.original_mean = np.array(kwargs['original_mean'])
        self.target_mean = np.array(kwargs['target_mean'])
        self.randomize = kwargs['randomize']
        original_dir = os.path.join(kwargs['data_root'], kwargs['original_font'])
        target_dir = os.path.join(kwargs['data_root'], kwargs['target_font'])
        start_index = kwargs['start_index']
        number = kwargs['number']

        # read data info from lists
        self.list_original = sorted(glob.glob(os.path.join(original_dir, '*.png')))[start_index : (start_index + number)]
        self.list_target = sorted(glob.glob(os.path.join(target_dir, '*.png')))[start_index : (start_index + number)]

        self.list_original = np.array(self.list_original, np.object)
        self.list_target = np.array(self.list_target, np.object)
        self.num = number
        print('# Images found:', self.num)

        # permutation
        np.random.seed()
        perm = np.random.permutation(self.num) 
        self.list_original[:] = self.list_original[perm, ...]
        self.list_target[:] = self.list_target[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        original_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 1))
        target_batch = np.zeros((batch_size, self.target_size, self.target_size, 1))
        for i in range(batch_size):
            original_image = scipy.misc.imread(self.list_original[self._idx])
            original_image = scipy.misc.imresize(original_image, (self.load_size, self.load_size))
            original_image = original_image.astype(np.float32)/255.
            # original_image = -(original_image - self.original_mean) # Revert to white characters on black background
            original_image = 1 - original_image

            target_image = scipy.misc.imread(self.list_target[self._idx])
            target_image = scipy.misc.imresize(target_image, (self.target_size, self.target_size))
            target_image = target_image.astype(np.float32)/255.
            # target_image = -(target_image - self.target_mean)
            target_image = 1 - target_image
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    original_image = original_image[:, ::-1]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            original_batch[i, :, :, 0] = original_image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]
            # target_batch[i, :, :] = target_image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]
            target_batch[i, :, :, 0] = target_image

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return original_batch, target_batch


    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
