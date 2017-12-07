import scipy.misc
from PIL import Image
import numpy as np
from os.path import join

OUT_DIR = '../../output'

def numpy_to_image(array2d):
    return Image.fromarray(np.uint8(array2d * 255), 'L')
    # return scipy.misc.toimage(array2d)


def show_comparison(original_batch, target_batch, result_batch, save=False, mode='', iter=-1):
    # print(result_batch[1, :, :])
    # print(original_batch.shape)
    batch_size, h, w, _ = target_batch.shape
    batch_size = min(10, batch_size)
    image = Image.new('L', (w * batch_size, h * 3))
    for i in range(batch_size):
        original_image = numpy_to_image(original_batch[i, :, :, 0])
        image.paste(original_image.resize((w, h)), (i * w, 0)) # Not sure about the sequece of w, h
        image.paste(numpy_to_image(result_batch[i, :, :, 0]), (i * w, h))
        image.paste(numpy_to_image(target_batch[i, :, :, 0]), (i * w, 2 * h))
    if save:
        assert iter != -1
        if mode in ['display_train', 'display_val']:
            affix = mode[8:]
            output_filename = 'comparison_iter=%04d_%s.png' % (iter, affix)
        elif mode is 'validate':
            output_filename = 'validation_%04d.png' % iter
        else:
            raise RuntimeError('mode could only be display_train, display_val or validate')
        image.save(join(OUT_DIR, output_filename))
    else:
        image.show()
