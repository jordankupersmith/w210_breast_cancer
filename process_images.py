from PIL import Image
import os
import re
import sys
import numpy as np
from tqdm import tqdm

import boto3


def process_image(res, cos, img_file, exceptions_dir, slice_size=128, offset=.5, positive_threshold = .1):
    try:
        res.Bucket('ddsm-clahe').download_file(img_file, img_file)
    except:
        print("couldn't download: " + img_file)

    im = Image.open(img_file)
    masks = [obj.key for obj in res.Bucket('ddsm-clahe').objects.all() if
             img_file[:-4] in obj.key and re.search('_\d.png', obj.key)]

    for mask in masks:
        try:
            res.Bucket('ddsm-clahe').download_file(mask, mask)
        except:
            print("couldn't download: " + img_file)

    true_mask = np.zeros(shape=im.size, dtype=np.uint8)
    try:
        for mask in masks:
            mask_img = np.array(Image.open(mask).getdata()).reshape(im.size[1], im.size[0]).T
            true_mask = np.logical_or(mask_img, true_mask).astype(np.uint8)
            # Image.fromarray(true_mask).show()
            # print mask_img.dtype
    except ValueError:
        print('mismatched mask size for: {}'.format(img_file))
        os.rename(img_file, '../{}/{}'.format(exceptions_dir, img_file))
        for mask in masks:
            os.rename(mask, '../{}/{}'.format(exceptions_dir, mask))
        return

    width, height = im.size
    pic = 1  # ?
    window_slide = slice_size * (1 - offset)
    map_img = Image.fromarray(true_mask.T.astype(np.uint8) * 255)
    for i in range(int((width / window_slide) - 1)):
        for j in range(int((height / window_slide) - 1)):
            box = (i * window_slide, j * window_slide, i * window_slide + slice_size, j * window_slide + slice_size)
            region = im.crop(box)
            region_map = map_img.crop(box)
            positive = False
            if sum(region_map.getdata()) > positive_threshold * 255 * slice_size**2:
                positive = True
            if positive == True:
                region.save('{}_{}_{}.png'.format(img_file[:-4], str(pic).zfill(5), 1))
                # region_map.save('{}_{}_{}_map.png'.format(img_file[:-4], str(pic).zfill(5), 1))
            else:
                if np.average(np.array(region.getdata()))>4:
                    region.save('{}_{}_{}.png'.format(img_file[:-4], str(pic).zfill(5), 0))

            pic += 1

    os.remove(img_file)
    for mask in masks:
        os.remove(mask)
    im.close()
    # for img in os.listdir('.'):
    #     cos.upload_file(img, 'ddsm-v2', img)
    #     os.remove(img)


if __name__ == "__main__":

    imgs = raw_input('In which directory do you want to put the images(defalut:"images")?\n')
    if imgs == '':
        imgs = 'images'
    os.chdir(imgs)

    exceptions_dir = raw_input('Where do you want the exceptions(defalut:"exceptions")?:\n')
    if exceptions_dir == '':
        exceptions_dir = 'exceptions'

    slice_size = raw_input('how large do you want your images in nxn(defalut:128)?: n=')
    if slice_size == '':
        slice_size = 128
    else:
        slice_size = int(slice_size)

    offset = raw_input('How much overlap should there be between images?(between 0 and 1)(defalut:.5)')
    if offset == '':
        offset = .5
    else:
        offset = float(offset)

    imgs_to_dnld = raw_input('How many images do you want to download?(defalut:10)')
    if imgs_to_dnld == '':
        imgs_to_dnld = 10
    else:
        imgs_to_dnld = int(imgs_to_dnld)
    # try:
    #     exceptions_dir = sys.argv[2]
    # except IndexError:
    #     exceptions_dir = 'exceptions'
    # try:
    #     slice_size = int(sys.argv[3])
    # except IndexError:
    #     slice_size = 128
    #
    # try:
    #     offset = float(sys.argv[4])
    # except IndexError:
    #     offset = .5
    #
    # try:
    #     imgs_to_dnld = int(sys.argv[5])
    # except IndexError:
    #     imgs_to_dnld = 50


    endpoint = 'https://s3-api.us-geo.objectstorage.softlayer.net'

    cos = boto3.client('s3', endpoint_url=endpoint)
    res = boto3.resource('s3', endpoint_url=endpoint)
    # if file not in bucket THEN upload

    # Find the images in the dataset that didn't convert, call them exceptions
    ddsm_objects = [obj.key for obj in res.Bucket('ddsm').objects.all()]
    ddsm_png_objects = [obj.key for obj in res.Bucket('ddsm-png').objects.all()]

    exceptions = [x for x in ddsm_objects if x[:-4] + '.png' not in ddsm_png_objects]
    final_excpetions = []
    for exception in exceptions:
        if re.search('_\d.dcm', exception):
            final_excpetions.append(exception[:-6] + '.png')
        else:
            final_excpetions.append(exception[:-4] + '.png')

    # Get all images from ddsm-clahe
    ddsm_clahe_objects = [obj.key for obj in res.Bucket('ddsm-clahe').objects.all()]

    # filter out all mask images
    ddsm_clahe_objects = [x for x in ddsm_clahe_objects if not re.search('_\d.png', x)]

    # filter out all images that were in exceptions
    ddsm_clahe_objects = [x for x in ddsm_clahe_objects if x not in final_excpetions]

    # filter out any objects that have already been converted
    # cut_objects = set([obj.key[:-12] for obj in res.Bucket('ddsm-v2').objects.all()])
    cut_objects = set([x[:-12] + '.png' for x in os.listdir('.')])
    objects = [x for x in ddsm_clahe_objects if x not in cut_objects]


    for img_file in tqdm(objects[:imgs_to_dnld]):
        # print img_file
        if img_file not in cut_objects:
            # print img_file
            process_image(res, cos, img_file, exceptions_dir, slice_size=slice_size, offset=offset)
