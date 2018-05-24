import numpy as np
import math
import os
import PIL
import xml.etree.ElementTree as ET

# get file name iterator
def get_file_list(img_dir, img_suffix='.jpg'):
	f_all_itr = (f for f in os.listdir(img_dir))
	f_itr = filter(lambda f:f.endswith(img_suffix), sorted(f_all_itr))
	f_itr = map(lambda f:f.split('.',1)[0], f_itr)
	return f_itr

# crop large original image into some smaller images with a n*m grid. corresponding annotation files are also generated.
def crop_img_2d(img_path, ant_path, out_path, sub_array=[3, 2]):
	img_ori = PIL.Image.open(img_path)
	img_size = img_ori.size
	grid_array = ([0]+[math.floor((i+1)*img_size[0]/float(sub_array[0])) for i in range(sub_array[0])], [0]+[math.floor((i+1)*img_size[1]/float(sub_array[1])) for i in range(sub_array[1])])
	get_sub_path = lambda sub_index, ori_path, out_path:out_path + '/' + os.path.splitext(os.path.basename(ori_path))[0] + '-%d'%(sub_index) + os.path.splitext(os.path.basename(ori_path))[1]
	ant_info, template_tree = _load_ant_file(ant_path)
	sub_index = 0
	for sub_index_x in range(sub_array[0]):
		for sub_index_y in range(sub_array[1]):
			sub_img_path = get_sub_path(sub_index, img_path, out_path)
			sub_ant_path = get_sub_path(sub_index, ant_path, out_path)
			x_min_img = grid_array[0][sub_index_x]
			y_min_img = grid_array[1][sub_index_y]
			x_max_img = grid_array[0][sub_index_x+1]-1
			y_max_img = grid_array[1][sub_index_y+1]-1
			img_box = np.array([x_min_img, y_min_img, x_max_img, y_max_img])
			objs_info = _prune_bndbox(ant_info, img_box)
			_save_sub_ant_file(sub_ant_path, template_tree, objs_info)
			sub_img = img_ori.crop((x_min_img, y_min_img, x_max_img+1, y_max_img+1))
			sub_img.save(sub_img_path)
			sub_index += 1

# load annotation file and return objects information
def _load_ant_file(ant_path):
	tree = ET.parse(ant_path)
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

# transfer box coordinates to the new image space, and prune unquanlified bounding boxes for cropped images.
def _prune_bndbox(objs_info, img_box):
	# transfer coordinate space
	x_ori = img_box[0]
	y_ori = img_box[1]
	img_width = img_box[2] - img_box[0] + 1
	img_height = img_box[3] - img_box[1] + 1
	objs_info['boxes'].astype(np.float32)
	new_boxes = np.zeros(objs_info['boxes'].shape, dtype=np.float32)
	new_boxes[:, 0] = objs_info['boxes'][:, 0] - x_ori
	new_boxes[:, 2] = objs_info['boxes'][:, 2] - x_ori
	new_boxes[:, 1] = objs_info['boxes'][:, 1] - y_ori
	new_boxes[:, 3] = objs_info['boxes'][:, 3] - y_ori

	# prune boxes
	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	gt_classes = []
	ishards = np.empty(shape=[0], dtype=np.int32)
	for i in range(len(objs_info['boxes'])):
		x1 = new_boxes[i, 0]
		y1 = new_boxes[i, 1]
		x2 = new_boxes[i, 2]
		y2 = new_boxes[i, 3]
		remove_flag, new_box = _bndbox_remove_flag(box=[x1, y1, x2, y2], shape_img=[img_width, img_height], ratio_lim=2.5, IOU_lim=0.55)
		if remove_flag:
		    continue

		difficult = objs_info['gt_ishard'][i]
		ishards = np.append(ishards, difficult)
		boxes = np.append(boxes, np.expand_dims(new_box, axis=0), axis=0)
		gt_classes.append(objs_info['gt_classes_name'][i])

	return {'boxes': boxes,
			'gt_classes_name': gt_classes,
			'gt_ishard': ishards}

def _bndbox_remove_flag(box, shape_img, ratio_lim, IOU_lim):
	new_box = np.zeros(4, dtype=np.uint16)
	new_box[0] = _new_cord(box[0], shape_img[0])
	new_box[1] = _new_cord(box[1], shape_img[1])
	new_box[2] = _new_cord(box[2], shape_img[0])
	new_box[3] = _new_cord(box[3], shape_img[1])
	IOU = (new_box[3]-new_box[1]+1)*(new_box[2]-new_box[0]+1)/((box[3]-box[1]+1)*(box[2]-box[0]+1))
	ratio = max(new_box[3]-new_box[1]+1, new_box[2]-new_box[0]+1)/min(new_box[3]-new_box[1]+1, new_box[2]-new_box[0]+1)
	return new_box[0]>=new_box[2] or new_box[1]>=new_box[3] or ratio>ratio_lim or IOU<IOU_lim, new_box

def _new_cord(cord, length):
	func_in_flag = lambda cord, length: 0<=cord and cord<length
	if func_in_flag(cord, length):
		new_cord = cord
	elif cord<0:
		new_cord = 0
	else:
		new_cord = length-1
	return new_cord


# save annotation file for cropped images.
def _save_sub_ant_file(file_path, template_tree, objs_info):
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


if __name__ == '__main__':

	ant_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/train_raw_review'
	img_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/train_raw_review'
	out_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/train'
	file_itr = get_file_list(img_dir=img_dir, img_suffix='.jpg')
	
	for file_name in file_itr:
		ant_path = ant_dir+'/'+file_name+'.xml'
		img_path = img_dir+'/'+file_name+'.jpg'
		crop_img_2d(img_path, ant_path, out_dir)