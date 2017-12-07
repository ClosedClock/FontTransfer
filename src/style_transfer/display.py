import scipy.misc
from PIL import Image
import numpy as np

def numpy_to_image(array2d):
    return scipy.misc.toimage(array2d)


def show_comparison(original_batch, target_batch, result_batch):
    # print(result_batch[1, :, :])
    # print(original_batch.shape)
    batch_size, h, w, _ = target_batch.shape
    batch_size = min(10, batch_size)
    image = Image.new('L', (w * batch_size, h * 3))
    for i in range(batch_size):
        original_image = numpy_to_image(original_batch[i, :, :, 0])
        # original_image = scipy.misc.imresize(original_image, [w, h]) # Not sure about the sequece of w, h
        image.paste(original_image, (i * w, 0))
        image.paste(numpy_to_image(result_batch[i, :, :, 0]), (i * w, h))
        image.paste(numpy_to_image(target_batch[i, :, :, 0]), (i * w, 2 * h))
    image.show()
