from PIL import Image


def process_image(img_file, mask_file):

	im =Image.open(img_file)
	mask = Image.open(mask_file)
	slice_size = 256

	width, height = im.size
	pic = 1
	for i in range(int(width/slice_size)):
		for j in range(int(height/slice_size)):
			box = (i*slice_size, j*slice_size, i*slice_size+slice_size, j*slice_size+slice_size)
			region = im.crop(box)
			region_map = mask.crop(box)
			positive = False
			for pixel in region_map.getdata():
				if pixel > 0 :
					positive = True
					break
			if positive == True:
				region.save('PNG/tmp/{}_{}.png'.format(1,pic))
				region_map.save('PNG/tmp/{}_{}_map.png'.format(1,pic))
			else:
				region.save('PNG/tmp/{}_{}.png'.format(0,pic))
			pic += 1


