#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont, ImageDraw
from os.path import join, exists
import os

## -------------------- Settings -----------------
_override = True # Control whether to override existing images
FONTS_DIR = join('..', 'fonts') # Folder where font files are saved
OUT_DIR = join('..', 'img') # Folder to save output images

# List of font files to be used in generating images
#font_files = [file for file in os.listdir(FONTS_DIR) if '.tt' in file] # all fonts
all_font_files = ['Baoli.ttc', 'Hannotate.ttc', 'Hanzipen.ttc', 'simkai.ttf', 'Songti.ttc',
                  'STHeiti.ttc', 'WeibeiSC.otf', 'Xingkai.ttc', 'Yuanti.ttc']
font_files = all_font_files

# List of font files that require an offset
offset_fonts = ['Baoli.ttc', 'Hannotate.ttc', 'WeibeiSC.otf', 'Hanzipen.ttc', 'Songti.ttc',
                'Xingkai.ttc', 'Yuanti.ttc']

COMMON_UNICODE_NUM = 0 # 0 for all common unicodes

## -------------------- Constants -----------------
SIZE = 80 # Size of image
START_INDEX = 0x4E00 # Unicode range of Chinese characters
END_INDEX = 0x9FBB

## -------------------- Main Program -----------------

def get_common_chinese_unicodes():
    '''
    Read saved for unicodes of common chinese characters and output it as
    a list.
    '''
    import csv
    common_chinese_unicodes = []
    with open(join('..', 'data', 'common_chinese_unicodes.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for line in spamreader:
            common_chinese_unicodes.extend(map(lambda x: int(x, 16), line[:-1]))
    return common_chinese_unicodes


def main():
    if COMMON_UNICODE_NUM is 0:
        unicodes = get_common_chinese_unicodes()
    else:
        unicodes = get_common_chinese_unicodes()[0:COMMON_UNICODE_NUM]

    if not exists(OUT_DIR):
        print('Warning: %s not exist' % OUT_DIR)
        os.makedirs(OUT_DIR)

    for font_file in font_files:
        font_name = font_file[:-4]
            
        if not exists(join(FONTS_DIR, font_file)):
            raise IOError('Font file not found: ' + font_file)
        
        if not exists(join(OUT_DIR, font_name)):
            os.makedirs(join(OUT_DIR, font_name))
            
        print('Current font: ' + font_file)
        font = ImageFont.truetype(join(FONTS_DIR, font_file), SIZE)
        for index in range(len(unicodes)):
            output_filename = join(OUT_DIR, font_name, font_name + '_%04d.png' % index)
            if exists(output_filename) and not _override:
                continue
            im = Image.new("L", (SIZE, SIZE), color=255) # 8-bit black and white
            draw = ImageDraw.Draw(im)
            if font_file in offset_fonts:
                draw.text((0, -20), chr(unicodes[index]), font=font)
            else:
                draw.text((0, 0), chr(unicodes[index]), font=font)
            # write into file
            im.save(output_filename)
            if (index + 1) % 100 == 0:
                print('.', end='')
        print('')

if __name__ == '__main__':
    main()
