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
        original_batch = np.zeros((batch_size, self.fine_size, self.fine_size))
        target_batch = np.zeros((batch_size, self.fine_size, self.fine_size))
        for i in range(batch_size):
            original_image = scipy.misc.imread(self.list_original[self._idx])
            original_image = scipy.misc.imresize(original_image, (self.load_size, self.load_size))
            original_image = original_image.astype(np.float32)/255.
            # original_image = -(original_image - self.original_mean) # Revert to white characters on black background
            original_image = 1 - original_image

            target_image = scipy.misc.imread(self.list_target[self._idx])
            target_image = scipy.misc.imresize(target_image, (self.load_size, self.load_size))
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

            original_batch[i, :, :] = original_image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]
            target_batch[i, :, :] = target_image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return original_batch, target_batch


    def size(self):
        return self.num

    def reset(self):
        self._idx = 0


# Loading data from disk
class TestDataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = sorted(glob.glob(os.path.join(self.data_root, '*.jpg')))
        self.list_filenames = ['/'.join(image_path.split('/')[-2:]) for image_path in self.list_im]
        self.list_im = np.array(self.list_im, np.object)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, 3, self.fine_size, self.fine_size))
        filenames_batch = []
        for i in range(batch_size):
            filenames_batch.append(self.list_filenames[self._idx])
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32) / 255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip > 0:
                    image = image[:, ::-1, :]
                offset_h = np.random.random_integers(0, self.load_size - self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size - self.fine_size)
            else:
                offset_h = (self.load_size - self.fine_size) // 2
                offset_w = (self.load_size - self.fine_size) // 2

            images_batch[i, 0, :, :] = image[offset_h:offset_h + self.fine_size, offset_w:offset_w + self.fine_size, 0]
            images_batch[i, 1, :, :] = image[offset_h:offset_h + self.fine_size, offset_w:offset_w + self.fine_size, 1]
            images_batch[i, 2, :, :] = image[offset_h:offset_h + self.fine_size, offset_w:offset_w + self.fine_size, 2]

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return torch.FloatTensor(images_batch), filenames_batch

    def size(self):
        return self.num

    def reset(self):
        self._idx = 0