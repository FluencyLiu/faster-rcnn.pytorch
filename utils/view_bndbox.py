import numpy as np
import os
import PIL
from PIL import ImageDraw
import xml.etree.ElementTree as ET
import general as ut

# view bounding box in test images.
def view_bndboxes_2d(img_path, ant_path, out_path):
	img = PIL.Image.open(img_path)
	boxes = _load_annt_file(ant_path)
	if boxes is None:
		img.save(out_path)
		return

	draw = ImageDraw.Draw(img)
	for i in range(boxes.shape[0]):
		[xmin,ymin,xmax,ymax] = boxes[i, :]
		draw.rectangle([xmin,ymin,xmax,ymax], outline=(0,255,0))
		draw.rectangle([xmin+1,ymin+1,xmax-1,ymax-1], outline=(0,255,0))
		draw.rectangle([xmin+2,ymin+2,xmax-2,ymax-2], outline=(0,255,0))
	img.save(out_path)
	return

# load annotation file and return bounding boxes
def _load_annt_file(ant_path):
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
	
	# ant_dir = '/home/lc/code/NetKit/data/output/restore/test'
	# img_dir = '/home/lc/code/NetKit/data/own_data/test_raw'
	# out_dir = '/home/lc/code/NetKit/data/output/visualize'

	# file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')

	# for file_name in file_itr:
	# 	ant_path = ant_dir+'/'+file_name+'.xml'
	# 	img_path = img_dir+'/'+file_name+'.jpg'
	# 	out_path = out_dir+'/'+file_name+'.jpg'
	# 	print(img_path)
	# 	view_bndboxes_2d(img_path, ant_path, out_path)

	# show test cases ==================================
	gt_ant_dir = '/home/lc/code/NetKit/data/own_data/test_raw'
	restore_ant_dir = '/home/lc/code/NetKit/data/output/restore/test'
	refine_ant_dir = '/home/lc/code/NetKit/data/output/refined/test'
	img_dir = '/home/lc/code/NetKit/data/own_data/test_raw'
	out_dir = '/home/lc/code/NetKit/data/output/visualize'

	file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')

	for file_name in file_itr:
		img_path = img_dir+'/'+file_name+'.jpg'
		gt_ant_path =  gt_ant_dir+'/'+file_name+'.xml'
		restore_ant_path =  restore_ant_dir+'/'+file_name+'.xml'
		refine_ant_path =  refine_ant_dir+'/'+file_name+'.xml'
		gt_out_path = out_dir+'/'+file_name+'-0.jpg'
		restore_out_path = out_dir+'/'+file_name+'-1.jpg'
		refine_out_path = out_dir+'/'+file_name+'-2.jpg'
		print(img_path)
		view_bndboxes_2d(img_path, gt_ant_path, gt_out_path)
		view_bndboxes_2d(img_path, restore_ant_path, restore_out_path)
		view_bndboxes_2d(img_path, refine_ant_path, refine_out_path)

	# show all_neg test cases ==================================
	restore_ant_dir = '/home/lc/code/NetKit/data/output/restore/all_neg_test'
	refine_ant_dir = '/home/lc/code/NetKit/data/output/refined/all_neg_test'
	img_dir = '/home/lc/code/NetKit/data/own_data/all_neg/test_raw'
	out_dir = '/home/lc/code/NetKit/data/output/visualize/all_neg'

	file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')

	for file_name in file_itr:
		img_path = img_dir+'/'+file_name+'.jpg'
		restore_ant_path =  restore_ant_dir+'/'+file_name+'.xml'
		refine_ant_path =  refine_ant_dir+'/'+file_name+'.xml'
		restore_out_path = out_dir+'/'+file_name+'-1.jpg'
		refine_out_path = out_dir+'/'+file_name+'-2.jpg'
		print(img_path)		
		view_bndboxes_2d(img_path, restore_ant_path, restore_out_path)
		view_bndboxes_2d(img_path, refine_ant_path, refine_out_path)