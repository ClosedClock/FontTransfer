from os.path import join
from PIL import Image, ImageFont, ImageDraw

string = '机器学习随便搞搞就行了'

## -------------------- Settings -----------------
font_files = ['Baoli.ttc', 'Hannotate.ttc', 'Hanzipen.ttc', 'simkai.ttf', 'Songti.ttc',
                  'STHeiti.ttc', 'WeibeiSC.otf', 'Xingkai.ttc', 'Yuanti.ttc']
FONTS_DIR = '../fonts' # Folder where font files are saved
OUT_DIR = '../fig/sample_characters' # Folder to save output images
SIZE = 80

offset_fonts = ['Baoli.ttc', 'Hannotate.ttc', 'WeibeiSC.otf', 'Hanzipen.ttc', 'Songti.ttc',
                'Xingkai.ttc', 'Yuanti.ttc']


def draw_characters_with_unicodes(string):
    for font_file in font_files:
        font_name = font_file[:-4]
        output_filename = join(OUT_DIR, font_name + '_display.png')
        im = Image.new("L", (SIZE * len(string), SIZE), color=0) # 8-bit black and white
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(join(FONTS_DIR, font_file), SIZE)
        offset = -20 if font_file in offset_fonts else 0
        draw.text((0, offset), string, font=font, fill=255)
        # write into file
        im.save(output_filename)


if __name__ == '__main__':
    draw_characters_with_unicodes(string)
