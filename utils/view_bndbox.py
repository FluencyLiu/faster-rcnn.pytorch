import numpy as np
import os
import PIL
from PIL import ImageDraw
import xml.etree.ElementTree as ET

# get file name iterator
def get_file_list(img_dir, img_suffix='.jpg'):
	f_all_itr = (f for f in os.listdir(img_dir))
	f_itr = filter(lambda f:f.endswith(img_suffix), sorted(f_all_itr))
	f_itr = map(lambda f:f.split('.',1)[0], f_itr)
	return f_itr

# view bounding box in test images.
def view_bndboxes_2d(img_path, ant_path, out_path):
	img = PIL.Image.open(img_path)
	boxes = _load_ant_file(ant_path)
	if boxes is None:
		img.save(out_path)
		return

	draw = ImageDraw.Draw(img)
	for i in range(boxes.shape[0]):
		[xmin,ymin,xmax,ymax] = boxes[i, :]
		draw.rectangle([xmin,ymin,xmax,ymax], outline=(0,255,0))
	img.save(out_path)
	return

# load annotation file and return bounding boxes
def _load_ant_file(ant_path):
	tree = ET.parse(ant_path)
	objs = tree.findall('object')
	num_objs = len(objs)
	if num_objs <= 0:
		return None
	boxes = np.empty(shape=[num_objs, 4], dtype=np.uint16)
	
	for ix, obj in enumerate(objs):
		bbox = obj.find('bndbox')
		x1 = int(bbox.find('xmin').text)
		y1 = int(bbox.find('ymin').text)
		x2 = int(bbox.find('xmax').text)
		y2 = int(bbox.find('ymax').text)
		boxes[ix, :] = [x1, y1, x2, y2]

	return boxes

if __name__ == '__main__':
	file_itr = get_file_list(img_dir='/home/lc/code/faster-rcnn.pytorch/data/own_data/test', img_suffix='.jpg')
	ant_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/test'
	img_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/test'
	out_dir = '/home/lc/code/faster-rcnn.pytorch/images/bndbox'
	for file_name in file_itr:
		ant_path = ant_dir+'/'+file_name+'.xml'
		img_path = img_dir+'/'+file_name+'.jpg'
		out_path = out_dir+'/'+file_name+'-gt.jpg'
		print(img_path)
		view_bndboxes_2d(img_path, ant_path, out_path)