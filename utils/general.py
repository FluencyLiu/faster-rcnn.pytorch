import numpy as np
import math
import os
import xml.etree.ElementTree as ET

# get file name iterator
def get_file_list(img_dir, img_suffix='.jpg'):
	f_all_itr = (f for f in os.listdir(img_dir))
	f_itr = filter(lambda f:f.endswith(img_suffix), sorted(f_all_itr))
	f_itr = map(lambda f:f.split('.',1)[0], f_itr)
	return f_itr

# load annotation file and return objects information
def load_annt_file(annt_path):
	tree = ET.parse(annt_path)
	objs = tree.findall('object')
	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	gt_classes = []
	ishards = np.empty(shape=[0], dtype=np.int32)
	for obj in objs:
		bbox = obj.find('bndbox')
		x1 = float(bbox.find('xmin').text) - 1 if float(bbox.find('xmin').text)>0 else 0
		y1 = float(bbox.find('ymin').text) - 1 if float(bbox.find('ymin').text)>0 else 0
		x2 = float(bbox.find('xmax').text) - 1
		y2 = float(bbox.find('ymax').text) - 1
		if x1 >= x2 or y1>=y2:
		    continue

		diffc = obj.find('difficult')
		difficult = 0 if diffc == None else int(diffc.text)
		ishards = np.append(ishards, difficult).astype(np.int32)
		# class_name = obj.find('name').text.lower().strip()
		# the following line is used for dirty data
		class_name = 'abnormal'
		boxes = np.append(boxes, np.expand_dims([x1, y1, x2, y2], axis=0).astype(np.uint16), axis=0)
		gt_classes.append(class_name)

	return {'boxes': boxes,
    		'gt_classes_name': gt_classes,
    		'gt_ishard': ishards}, tree

# save annotation file for cropped images.
def save_annt_file(file_path, objs_info, template_tree=None):
	if template_tree == None:
		root = ET.Element('annotation')
		folder = ET.SubElement(root, 'folder')
		folder.text = 'undefined'
		filename = ET.SubElement(root, 'filename')
		filename.text = 'undefined'
		path = ET.SubElement(root, 'path')
		path.text = 'undefined'
		source = ET.SubElement(root, 'source')
		database = ET.SubElement(source, 'database')
		database.text = 'unknown'
		size = ET.SubElement(root, 'size')
		width = ET.SubElement(size, 'width')
		height = ET.SubElement(size, 'height')
		depth = ET.SubElement(size, 'depth')
		width.text = str(0)
		height.text = str(0)
		depth.text = str(0)
		segmented = ET.SubElement(root, 'segmented')
		segmented.text = str(0)
	else:
		root = template_tree.getroot()
		objs = template_tree.findall('object')
		for obj in objs:
			root.remove(obj)

	for i in range(len(objs_info['gt_classes_name'])):
		obj = ET.SubElement(root, 'object')
		name = ET.SubElement(obj, 'name')
		name.text = objs_info['gt_classes_name'][i]
		pose = ET.SubElement(obj, 'pose')
		pose.text = 'Unspecified'
		truncated = ET.SubElement(obj, 'truncated')
		truncated.text = str(0)
		difficult = ET.SubElement(obj, 'difficult')
		difficult.text = str(objs_info['gt_ishard'][i])
		bndbox = ET.SubElement(obj, 'bndbox')
		xmin = ET.SubElement(bndbox, 'xmin')
		ymin = ET.SubElement(bndbox, 'ymin')
		xmax = ET.SubElement(bndbox, 'xmax')
		ymax = ET.SubElement(bndbox, 'ymax')
		xmin.text = str(int(objs_info['boxes'][i,0]))
		ymin.text = str(int(objs_info['boxes'][i,1]))
		xmax.text = str(int(objs_info['boxes'][i,2]))
		ymax.text = str(int(objs_info['boxes'][i,3]))
	
	tree = ET.ElementTree(root)
	tree.write(file_path, encoding='UTF-8')
