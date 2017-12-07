import scipy.misc
from PIL import Image
import numpy as np

def numpy_to_image(array2d):
    return scipy.misc.toimage(array2d)


def show_comparison(original_batch, target_batch, result_batch):
    # print(result_batch[1, :, :])
    batch_size, h, w = original_batch.shape
    batch_size = min(10, batch_size)
    image = Image.new('L', (w * batch_size, h * 3))
    for i in range(batch_size):
        image.paste(numpy_to_image(original_batch[i, :, :]), (i * w, 0))
        image.paste(numpy_to_image(result_batch[i, :, :]), (i * w, h))
        image.paste(numpy_to_image(target_batch[i, :, :]), (i * w, 2 * h))
    image.show()
