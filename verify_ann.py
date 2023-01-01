import argparse
import json
import pathlib

from PIL import Image
from PIL import ImageDraw

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--target', default='train.json')
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    data_dir = pathlib.Path('datasets')

    with open(data_dir / args.dataset / 'annotations' / args.target) as stream:
        contents = json.load(stream)
    images, annotations = contents['images'], contents['annotations']

    print('Images:', len(images), 'Annotations:', len(annotations))

    image, = filter(lambda di: di['id'] == args.index, images)
    *annotations, = filter(lambda di: di['image_id'] == args.index, annotations)

    im = Image.open(data_dir / args.dataset / image['file_name'])
    draw = ImageDraw.Draw(im)

    for annotation in annotations:
        left, top, width, height = annotation['bbox_vis']
        draw.rectangle(((left, top), (left + width, top + height)))

    im.show() if args.show else im.save('annotated.png')
