import torch
import glob 

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
from sklearn.preprocessing import scale


def parse_annotations(txt_paths):
    columns = ['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion']
    meta_data = pd.DataFrame(columns=columns)
    meta_data['height'] = None
    meta_data['width'] = None

    for txt_path in txt_paths:
        df = pd.read_csv(txt_path, names=columns)
        img_name = txt_path.split('/')[-1].split('.')[0]
        df.index = [img_name]*len(df)
        im = cv2.imread(BASE_IMAGE + img_name + '.jpg')
        height, width, _ = im.shape
        df['height'] = height
        df['width'] = width
        meta_data = meta_data.append(df)
    return meta_data


def rescale(s):
    n_w = [(s.width - s.height)/2, (s.height + s.width)/2]
    if s.bbox_left < n_w[0] or s.bbox_left + s.bbox_width < n_w[0] or s.bbox_left > n_w[1] or s.bbox_left + s.bbox_width > n_w[1]:
        s.bbox_left = None
        s.bbox_top = None
        s.bbox_width = None
        s.bbox_height = None
    else:
        s.bbox_left = (s.bbox_left - n_w[0]) * 300 / s.height
        s.bbox_top = s.bbox_top  * 300 / s.height
        s.bbox_width = s.bbox_width * 300 / s.height
        s.bbox_height = s.bbox_height * 300 / s.height
    
    return s


def iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def _main():
	precision = 'fp32'
	ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
	utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
	paths = glob.glob('./drive/My Drive/Dataset/Dataset_lab_2/VisDrone2019-DET-val/images/*.jpg')
	ssd_model.to('cuda')
	ssd_model.eval()
	print(f'Image count - {len(paths)}')
	txt_paths = glob.glob('./drive/My Drive/Dataset/Dataset_lab_2/VisDrone2019-DET-val/annotations/*.txt')

	BASE_IMAGE = "./drive/My Drive/Dataset/Dataset_lab_2/VisDrone2019-DET-val/images/"
	BASE_ANNOTATION = "./drive/My Drive/Dataset/Dataset_lab_2/VisDrone2019-DET-val/annotations/"

	classes ={0 :'ignored regions',
		      1 :'pedestrian',
		      2 :'people',
		      3 :'bicycle',
		      4 :'car',
		      5 :'van',
		      6 :'truck',
		      7 :'tricycle',
		      8 :'awning-tricycle',
		      9 :'bus',
		      10 :'motor',
		      11 :'others'}

	translate_classes ={0 : None,
		                1 :1,
		                2 :1,
		                3 :2,
		                4 :3,
		                5 :None,
		                6 :8,
		                7 :None,
		                8 :None,
		                9 :6,
		                10 :4,
		                11 :None}

	meta_data = parse_annotations(txt_paths)
	meta_data.to_csv('meta_data.csv')
	print(f"Croped bboxes - {meta_data_rescaled.bbox_left.isna().sum()}")
	meta_data_rescaled = meta_data_rescaled.dropna()
	meta_data_rescaled.object_category = meta_data_rescaled.object_category.replace(translate_classes)
	print(f"Unknown classes - {meta_data_rescaled.object_category.isna().sum()}")
	meta_data_rescaled = meta_data_rescaled.dropna()
	print(f"Image count - {len(meta_data_rescaled.index.unique())}")

	classes_to_labels = utils.get_coco_object_dictionary()

	inputs = [utils.prepare_input(path) for path in paths]
	tensor = utils.prepare_tensor(inputs, precision == 'fp16')	
	
	batchs = []
	results = []

	for i in np.arange(0, 548):
		with torch.no_grad():
		    detections_batch = ssd_model(tensor[i:i+1])
		    batchs.append(detections_batch)
		    try:
		        results_per_input = utils.decode_results(detections_batch)
		    except RuntimeError:
		        print(f'Failed to load {i}')
		    best_results_per_input = [utils.pick_best(results, 0.30) for results in results_per_input]
		    results.append(best_results_per_input)
	
	
	final_result = []
	idx = 0 
	for result in results:
	    image = paths[idx].split('/')[-1].split('.')[0]
	    idx+=1
	    if image not in meta_data_rescaled.index:
		print(f"Skip image - {image}")
		continue
	    result_bboxes = result[0][0] * 300
	    result_classes = result[0][1]
	    confidences = result[0][2]
	    true_boxes = meta_data_rescaled.loc[image][['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']].values
	    true_classes = meta_data_rescaled.loc[image][['object_category']].values
	    if len(true_classes) == 1:
		true_boxes = [true_boxes]
	    for true_idx in range(len(true_classes)):
		for result_idx in range(len(result_classes)):
		    IoU = iou(result_bboxes[result_idx], [true_boxes[true_idx][0],
		                                        true_boxes[true_idx][1],
		                                        true_boxes[true_idx][0] + true_boxes[true_idx][2],
		                                        true_boxes[true_idx][1] + true_boxes[true_idx][3]])
		    
		    final_result.append([idx, image, true_classes[true_idx], result_classes[result_idx], IoU, result_bboxes[result_idx], true_boxes[true_idx]])


	res = np.array(final_result)
	res = pd.DataFrame(res, columns=['id', 'image', 'true_class', 'result_class',  'iou', 'output_bboxes', 'true_bboxes'])
	res.true_class = res.true_class.apply(lambda s: s if type(s) == np.float64 else s[0])
	res.true_class = res.true_class.astype(int)
	res.true_bboxes = res.true_bboxes.astype(str)

	task_1 = pd.DataFrame(columns=['iou_threshold', 'accuracy', 'missed', 'false_detected'])
iou_thresholds = [0.5, 0.75, 0.95]


	for iou_threshold in iou_thresholds:
	    max_iou_per_true_bbox = res.groupby('true_bboxes')['iou'].max()
	    missed = max_iou_per_true_bbox.loc[max_iou_per_true_bbox < iou_threshold].shape[0]
	    bboxes_true_class = res.loc[res.true_class == res.result_class].groupby('true_bboxes')['iou'].max()
	    true_bboxes = bboxes_true_class.loc[bboxes_true_class > iou_threshold].shape[0]
	    full_bboxes = max_iou_per_true_bbox.shape[0]
	    accuracy = true_bboxes / full_bboxes
	    bboxes_false_class = res.loc[res.true_class != res.result_class].groupby('true_bboxes')['iou'].max()
	    false_detected = bboxes_false_class.loc[bboxes_false_class > iou_threshold].shape[0]
	    task_1 = task_1.append({'iou_threshold': iou_threshold,
		                    'accuracy': accuracy,
		                    'missed':missed,
		                    'false_detected':false_detected}, ignore_index=True)
	print(task_1)
	
	task_2 = pd.DataFrame(columns=['class', 'iou_threshold', 'accuracy', 'missed', 'false_detected'])
	classes = [1, 4, 3, 8, 2, 6]
	
	for current_class in classes:
	    for iou_threshold in iou_thresholds:
		max_iou_per_true_bbox = res.loc[res.true_class == current_class].groupby('true_bboxes')['iou'].max()
		missed = max_iou_per_true_bbox.loc[max_iou_per_true_bbox < iou_threshold].shape[0]
		bboxes_true_class = res.loc[(res.true_class == current_class)&(res.true_class == res.result_class)].groupby('true_bboxes')['iou'].max()
		true_bboxes = bboxes_true_class.loc[bboxes_true_class > iou_threshold].shape[0]
		full_bboxes = max_iou_per_true_bbox.shape[0]
		accuracy = true_bboxes / full_bboxes
		bboxes_false_class = res.loc[(res.true_class == current_class) & (res.true_class != res.result_class)].groupby('true_bboxes')['iou'].max()
		false_detected = bboxes_false_class.loc[bboxes_false_class > iou_threshold].shape[0]
		task_2 = task_2.append({'class': current_class,
		                        'iou_threshold': iou_threshold,
		                        'accuracy': accuracy,
		                        'missed':missed,
		                        'false_detected':false_detected}, ignore_index=True)
	task_2['class'] = task_2['class'].astype(int).apply(lambda s: classes_to_labels[s - 1])
	print(task_2)

	images = ['0000193_01705_d_0000112', '0000193_01876_d_0000113', '0000193_01497_d_0000111']
	image_idx = [0, 546, 547]

	for idx in range(len(image_idx)):
		fig, ax = plt.subplots(1)
		image = inputs[image_idx[idx]] / 2 + 0.5
		ax.imshow(image)
		output_bboxes = df[(df.image == images[idx]) & (df.iou > 0)].output_bboxes
		true_bboxes = df[(df.image == images[idx]) & (df.iou > 0)].true_bboxes
		for bbox_id in range(len(true_bboxes)):
		    x, y, w, h = true_bboxes.iloc[bbox_id]
		    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
		    ax.add_patch(rect)

		for bbox_id in range(len(output_bboxes)):
		    left, bot, right, top = output_bboxes.iloc[bbox_id]
		    x, y, w, h = [left, bot, right - left, top - bot]
		    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
		    ax.add_patch(rect)
		

	plt.show()

if __name__ == '__main__':
	_main()
