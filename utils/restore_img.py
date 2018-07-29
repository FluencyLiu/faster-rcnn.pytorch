import numpy as np
import math
import os
import PIL
import general as ut

# restore some smaller images with a n*m grid. corresponding annotation files are also generated.
def _restore_img_2d(img_path, input_dir, output_dir, sub_array=[3, 2], img_suffix='.jpg'):
	get_sub_path = lambda sub_index, file_name, output_dir, file_suffix:output_dir + '/' + file_name + '-%d'%(sub_index) + file_suffix

	file_name = os.path.splitext(os.path.basename(img_path))[0]
	img_ori = PIL.Image.open(img_path)
	img_size = img_ori.size
	grid_array = ([0]+[math.floor((i+1)*img_size[0]/float(sub_array[0])) for i in range(sub_array[0])], [0]+[math.floor((i+1)*img_size[1]/float(sub_array[1])) for i in range(sub_array[1])])

	boxes = np.empty(shape=[0, 4], dtype=np.uint16)
	gt_classes = [];
	ishards = np.empty(shape=[0], dtype=np.int32)

	sub_index = 0
	for sub_index_x in range(sub_array[0]):
		for sub_index_y in range(sub_array[1]):
			sub_annt_path = get_sub_path(sub_index, file_name, input_dir, '.xml')
			sub_index += 1
			x_min_img = grid_array[0][sub_index_x]
			y_min_img = grid_array[1][sub_index_y]
			x_max_img = grid_array[0][sub_index_x+1]-1
			y_max_img = grid_array[1][sub_index_y+1]-1
			img_box = np.array([x_min_img, y_min_img, x_max_img, y_max_img])

			objs_info, template_tree = ut.load_annt_file(sub_annt_path)
			objs_info_restored = _restore_cord(objs_info, img_box)
			if objs_info_restored==None:
				continue

			boxes = np.append(boxes, objs_info['boxes'], axis=0)
			ishards = np.append(ishards, objs_info['gt_ishard'], axis=0)
			gt_classes.extend(objs_info['gt_classes_name'])
			
	objs_info = {'boxes': boxes,
				'gt_classes_name': gt_classes,
				'gt_ishard': ishards}
	annt_path = output_dir + '/' + file_name + '.xml'
	ut.save_annt_file(annt_path, objs_info)

# box coordinates transformation of boxes
def _restore_cord(objs_info, img_box):
	if len(objs_info['gt_classes_name']) == 0:
		return None

	x_ori = img_box[0]
	y_ori = img_box[1]
	new_boxes = np.zeros(objs_info['boxes'].shape, dtype=np.uint16)
	new_boxes[:, 0] = objs_info['boxes'][:, 0] + x_ori
	new_boxes[:, 2] = objs_info['boxes'][:, 2] + x_ori
	new_boxes[:, 1] = objs_info['boxes'][:, 1] + y_ori
	new_boxes[:, 3] = objs_info['boxes'][:, 3] + y_ori
	objs_info['boxes'] = new_boxes
	return objs_info

if __name__ == '__main__':
	img_dir = '/home/lc/code/faster-rcnn.pytorch/data/own_data/test_0708_raw'
	input_dir = '/home/lc/code/faster-rcnn.pytorch/output/rcnn/test_0708'
	out_dir = '/home/lc/code/faster-rcnn.pytorch/output/restore/test_0708'
	file_itr = ut.get_file_list(img_dir=img_dir, img_suffix='.jpg')
	
	for file_name in file_itr:
		img_path = img_dir+'/'+file_name+'.jpg'
		print(img_path)
		_restore_img_2d(img_path, input_dir, out_dir)
