#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont, ImageDraw
from os.path import join, exists
import os

SIZE = 80 # Size of image
START_INDEX = 0x4E00 # Unicode range of Chinese characters
END_INDEX = 0x9FBB
OVERRIDE = True # Control whether to override existing images

def get_common_chinese_unicodes():
    import csv
    common_chinese_unicodes = []
    with open(join('..', 'data', 'common_chinese_unicodes.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for line in spamreader:
            common_chinese_unicodes.extend(map(lambda x: int(x, 16), line[:-1]))
    return common_chinese_unicodes


def main():
    unicodes = get_common_chinese_unicodes()[100:110]

    FONTS_DIR = join('..', 'fonts') # Folder where font files are saved
    OUT_DIR = join('..', 'test_output') # Folder to save output images

    if not exists(OUT_DIR):
        print('Warning: %s not exist' % OUT_DIR)
        os.makedirs(OUT_DIR)

    # List of font files to be used in generating images
    font_files = [file for file in os.listdir(FONTS_DIR) if '.tt' in file]
    #font_files = ['simkai.ttf', 'Xingkai.ttc']
    offset_fonts = ['Baoli.ttc', 'Hannotate.ttc', 'Hanzipen.ttc', 'Songti.ttc', 'Xingkai.ttc', 'Yuanti.ttc']

    for font_file in font_files:
        if not exists(join(FONTS_DIR, font_file)):
            raise IOError('Font file not found: ' + font_file)
        print('Current font: ' + font_file)
        font = ImageFont.truetype(join(FONTS_DIR, font_file), SIZE)
        for index in range(len(unicodes)):
            output_filename = join(OUT_DIR, font_file[:-4] + '_%04d.png' % index)
            if exists(output_filename) and not OVERRIDE:
                continue
            im = Image.new("L", (SIZE, SIZE), color=255) # 8-bit black and white
            draw = ImageDraw.Draw(im)
            if font_file in offset_fonts:
                draw.text((0, -20), chr(unicodes[index]), font=font)
            else:
                draw.text((0, 0), chr(unicodes[index]), font=font)
            # write into file
            im.save(output_filename)

if __name__ == '__main__':
    main()
