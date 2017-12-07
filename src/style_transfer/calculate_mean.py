import os, glob
import scipy.misc
import numpy as np

def calculate_mean():
    img_root = '../../img/'
    font_list = [dir_name for dir_name in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, dir_name))]
    mean_map = {}
    for font in font_list:
        print('Calculating mean for font: ' + font)
        img_list = glob.glob(os.path.join(img_root, font, '*.png'))
        mean_total = 0.
        for img_filename in img_list:
            image = scipy.misc.imread(img_filename)
            image = image.astype(np.float32)/255.
            mean_total += np.mean(image)
        mean_total /= len(img_list)
        mean_map[font] = mean_total

    return mean_map


if __name__ == '__main__':
    mean_map = calculate_mean()
    print(mean_map)