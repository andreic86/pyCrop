#!/usr/bin/env python3

# Simple image cropper based on person detection in the image
# So far tested only with jpeg images and python 3.9
# Works for me so must be working for you too :)

import argparse
import cv2
import os
from argparse import ArgumentParser
from datetime import datetime


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.gif', '.bmp']
AR_16x9 = 16 / 9
AR_9x16 = 9 / 16
AR_4x3 = 4 / 3
AR_3x4 = 3 / 4
AR_1x1 = 1 / 1


def parse_arguments():
    parser = ArgumentParser(description='CLI image cropper to predefined format with priority on keeping the people '
                                        'untouched.')
    parser.add_argument('--dir', required=False, nargs='*',
                        help='Directory to look for images to crop.')
    parser.add_argument('--input-files', required=False, nargs='*',
                        help='List of images to crop.')
    parser.add_argument('--out-dir', required=False,
                        help='Directory to save the modified images into. If not given, they will be saved beside the '
                             'original.')
    parser.add_argument('--crop', required=False, nargs=argparse.ONE_OR_MORE,
                        choices=['16x9', 'hdmi', '4x3', 'crt', '1x1', 'square'],
                        help='Saves a cropped copy of input image(s) in (out-)dir.')
    parser.add_argument('--resize', required=False, action='store_true', default=False,
                        help='If enabled, will resize photos to something smaller than 1000px on the shortest side')
    parser.add_argument('--no-detection', required=False, action='store_true', default=False,
                        help='Disables human detection in the input image(s) before cropping. '
                             'When detection disabled, cropping is done with focus on the middle of the image. '
                             'Otherwise cropping tries to best  fit the person.')
    parser.add_argument('--save-detection', required=False, action='store_true', default=False,
                        help='If detection enabled, saves a marked file with detections.')
    parser.add_argument('--debug', required=False, action='store_true', default=False,
                        help='Print debugging information during run. Not too much, not too little, not too '
                             'interesting either.')
    return parser.parse_args()


def list_images(in_dir, in_images):
    files = list()
    if in_dir is not None:
        for directory in in_dir:
            for root, _, found_files in os.walk(directory):
                for file in found_files:
                    if str(os.path.splitext(file)[1]).lower() in IMG_EXTENSIONS:
                        files.append(os.path.join(root, file))
    if in_images is not None:
        for file in in_images:
            if str(os.path.splitext(file)[1]).lower() in IMG_EXTENSIONS:
                files.append(file)
    return files


def get_col_crop(col, interest_area, crop_size, image_path, aspect):
    desired_size = col - crop_size
    interest_mid = int((interest_area[0] + interest_area[2]) / 2)
    if interest_area[0] <= 0:
        col_start, col_end = 0, desired_size
    elif interest_area[2] - interest_area[0] > desired_size:
        print(f"Possible person cropping in {image_path} for aspect {aspect}")
        col_start, col_end = interest_area[0], interest_area[0] + desired_size
    else:
        if interest_mid + desired_size / 2 >= col:
            col_start, col_end = crop_size, col
        elif interest_mid - desired_size / 2 <= 0:
            col_start, col_end = 0, desired_size
        else:
            col_start, col_end = interest_mid - desired_size / 2, interest_mid + desired_size / 2
    return col_start, col_end


def get_row_crop(row, interest_area, crop_size, image_path, aspect):
    desired_size = row - crop_size
    interest_mid = int((interest_area[1] + interest_area[3]) / 2)
    if interest_area[1] <= 0:
        row_start, row_end = 0, desired_size
    elif interest_area[3] - interest_area[1] > desired_size:
        print(f"Possible person cropping in {image_path} for aspect {aspect}")
        row_start, row_end = interest_area[1], interest_area[1] + desired_size
    else:
        if interest_mid + desired_size / 2 >= row:
            row_start, row_end = crop_size, row
        elif interest_mid - desired_size / 2 <= 0:
            row_start, row_end = 0, desired_size
        else:
            row_start, row_end = interest_mid - desired_size / 2, interest_mid + desired_size / 2
    return row_start, row_end


def get_cropping_area_horizontal(col, row, interest_area, image_path, aspect_ratio, cr):
    row_start, row_end = 0, row
    col_start, col_end = 0, col
    current_aspect = col / row
    if current_aspect == aspect_ratio:  # done
        pass
    elif current_aspect > aspect_ratio:  # crop left/right
        crop_size = int(col - row * aspect_ratio)
        col_start, col_end = crop_size / 2, col - crop_size / 2
        if interest_area:
            col_start, col_end = get_col_crop(col, interest_area, crop_size, image_path, cr)
    else:  # crop top/bottom
        crop_size = int(row - col * 1 / aspect_ratio)
        row_start, row_end = crop_size / 2, row - crop_size / 2
        if interest_area:
            row_start, row_end = get_row_crop(row, interest_area, crop_size, image_path, cr)

    return int(col_start), int(col_end), int(row_start), int(row_end)


def get_cropping_area_vertical(col, row, interest_area, image_path, aspect_ratio, cr):
    row_start, row_end = 0, row
    col_start, col_end = 0, col
    current_aspect = col / row
    if current_aspect == aspect_ratio:  # done
        pass
    elif current_aspect < aspect_ratio:  # crop top/bot
        crop_size = int(row - col * 1 / aspect_ratio)
        row_start, row_end = crop_size / 2, row - crop_size / 2
        if interest_area:
            row_start, row_end = get_row_crop(row, interest_area, crop_size, image_path, cr)
    else:  # crop left/right
        crop_size = int(col - row * aspect_ratio)
        col_start, col_end = crop_size / 2, col - crop_size / 2
        if interest_area:
            col_start, col_end = get_col_crop(col, interest_area, crop_size, image_path, cr)

    return int(col_start), int(col_end), int(row_start), int(row_end)


def crop_image(image_path, no_detection, out_dir, crop_request_list, resize, save_detection, debug):
    import cvlib
    if not out_dir:
        out_dir = os.path.dirname(image_path)
    read_image = cv2.imread(image_path)
    row, col, _ = read_image.shape

    interest_area = None
    bbox, labels, confidence = None, None, None

    if not no_detection:
        bbox, labels, confidence = cvlib.detect_common_objects(read_image, confidence=0.7, enable_gpu=False)
        if labels and 'person' in labels:
            if debug or len(crop_request_list) == 0:
                print(f"Found person(s) in {image_path}")
            index = 0
            for label in labels:
                if label == 'person':
                    if interest_area is None:
                        interest_area = bbox[index]
                    else:
                        interest_area = [min(interest_area[0], bbox[index][0]),
                                         min(interest_area[1], bbox[index][1]),
                                         max(interest_area[2], bbox[index][2]),
                                         max(interest_area[3], bbox[index][3])]
                index = index + 1
        elif debug:
            print(f"No person in {image_path} ({labels})")

    for cr in crop_request_list:
        if row < col:
            aspect_ratio = AR_1x1 if cr == '1x1' else AR_4x3 if cr == '4x3' else AR_16x9
            col_start, col_end, row_start, row_end = \
                get_cropping_area_horizontal(col, row, interest_area, image_path, aspect_ratio, cr)
        else:
            aspect_ratio = AR_1x1 if cr == '1x1' else AR_3x4 if cr == '4x3' else AR_9x16
            col_start, col_end, row_start, row_end = \
                get_cropping_area_vertical(col, row, interest_area, image_path, aspect_ratio, cr)

        if debug:
            print(f"Cropping [0:{col}, 0:{row}]:[{col_start}:{col_end}, {row_start}:{row_end}] ", end='')
            print(f"(interest_area {interest_area}).")
        cropped_image = read_image[row_start:row_end, col_start:col_end]
        if resize:
            scale_factor = col / 900 if col < row else row / 900
            new_size = (int((col_end - col_start) / scale_factor), int((row_end - row_start) / scale_factor))
            cropped_image = cv2.resize(cropped_image, new_size)
        save_cropped = os.path.basename(os.path.splitext(image_path)[0]) + f"_{cr}.jpeg"
        cv2.imwrite(out_dir + '/' + save_cropped, cropped_image)

    if save_detection and not no_detection:
        marked_image = cvlib.object_detection.draw_bbox(read_image, bbox, labels, confidence, write_conf=True)
        save_marked_path = os.path.basename(os.path.splitext(image_path)[0]) + '_detection.jpeg'
        cv2.imwrite(out_dir + '/' + save_marked_path, marked_image)
        if debug:
            print(f"Saved marked image with detection to {save_marked_path}")


if __name__ == '__main__':
    start_time = datetime.now()
    args = vars(parse_arguments())
    dbg = args['debug']
    images = list_images(args['dir'], args['input_files'])
    if dbg:
        print(f"Found a total of {len(images)} images to look at")
    find_files_time = datetime.now() - start_time
    start_time = datetime.now()

    if args['out_dir'] and not os.path.isdir(args['out_dir']):
        os.makedirs(args['out_dir'])
    crop_request = list()
    if args['crop']:
        if '16x9' in args['crop'] or 'hdmi' in args['crop']:
            crop_request.append('16x9')
        if '4x3' in args['crop'] or 'crt' in args['crop']:
            crop_request.append('4x3')
        if '1x1' in args['crop'] or 'square' in args['crop']:
            crop_request.append('1x1')
    for image in images:
        crop_image(image,
                   args['no_detection'],
                   args['out_dir'],
                   crop_request,
                   args['resize'],
                   args['save_detection'],
                   dbg)
    crop_time = datetime.now() - start_time
    print(f"It took {find_files_time} to find files and {crop_time} to crop {len(images)} images.")
