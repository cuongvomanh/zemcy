
'''
window = (left, top, w, h)
box = ((center_x, center_y), (a, b), angle) // or rect
((0.0, 0.0), (0.0, 0.0), 0.0)
(-angle) is angle of vecto of a is created by center to right and Ox

! points is np.array
! points la contour
box <-> points -> window <-> two points:
	cnt/points -> box: new_box = cv.minAreaRect(points) voi points la np.array
	box -> points: points = cv.boxPoints(box) ; points = np.int0(points)
	points -> window: cv.boundingRect(points) voi points la np.array
	box -> window: box->points->window

	window -> two points: two_points = window_to_two_points(window)
	two points -> window: window = two_points_to_window(two_points)

	boxs = contours_to_boxs(contours)
rotate:
	box-> points: convert_points_by_rotation_matrix(points, matrix)
	window > points

draw: 
	points: cv.polylines(img,points,True,(0,255,255))
	cv.drawContours(draw_resized_img, rect_contours, -1, (0, 255, 0), 1 )
	box: box -> points
	windows: draw_windows(img, windows, color, thickness)
numpy:
'''
# Python 2/3 compatibility
from __future__ import print_function, division
import sys
PY3 = sys.version_info[0] == 3
if PY3:
	xrange = range
import os
from glob import glob
import math
import numpy as np
import cv2 as cv
import statistics
import imutils
import random
from matplotlib import pyplot as plt
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm



DEBUG = False
VISION_SIZE = '480x270'
STANDARD_SIZE = '960x540'
STANDARD_AREA = 960*540
STANDARD_AREA_DICT = {'1': 1024*576, '2': 960*540, '3': 640*360}
global global_colors
global_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
# global_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
global global_color_index
global_color_index = 0


def points_to_box(points):
	return cv.minAreaRect(points)


def box_to_points(box):
	points = cv.boxPoints(box)
	points = np.int0(points)
	return points


def points_to_window(points):
	return cv.boundingRect(points)


def box_to_window(box):
	points = box_to_points(box)
	return points_to_window(points)
	

def window_to_two_points(window):
	left, top, w, h = window
	topleft = [left, top]
	bottomright = [left + w, top +h]
	return [topleft, bottomright]


def window_to_unit(window):
	left, top, w, h = window
	unit = (_center_x, _center_y), (_w, _h) = (left + w//2, top + h//2), (w, h)
	return unit


def two_points_to_window(points):
	topleft, bottomright = points
	left, top = topleft
	w, h = bottomright[0] - left, bottomright[1] - top
	return (left, top, w, h)


def add_padding_to_img(img, padding):
	width, height = _resolution = get_resolution(img)
	padding_width, padding_height = width + 2*padding, height + 2*padding
	padding_img = create_img((padding_width, padding_height))
	padding_img[padding: height + padding, padding: width + padding] = img
	return padding_img


def boxs_to_imgs_with_padding(img, boxs, is_w_bigger_than_h = True, padding = 5):
	padding_img = add_padding_to_img(img, padding)
	padding_box_imgs = []
	for box in boxs:
		(center_x, center_y), (a, b), angle = box
		padding_box = (center_x + padding, center_y + padding), (a + 2*padding, b+ 2*padding), angle
		padding_box_img = box_to_img(padding_img, padding_box, is_w_bigger_than_h)
		padding_box_imgs.append(padding_box_img)
	return padding_box_imgs


def box_to_img(img, box, is_w_bigger_than_h = True):
	img_width, img_height = get_resolution(img)
	(center_x, center_y), (a, b), angle = box
	(new_w, new_h) = (int(a), int(b)) if a>b  else (int(b), int(a))
	bounding_window = box_to_window(box)
	_left, _top, w, h = bounding_window
	simple_cut_img_w, simple_cut_img_h = max(new_w, w), max(new_h, h)
	padding = - int(min(center_x - simple_cut_img_w, img_width - center_x - 
			simple_cut_img_w, center_y - simple_cut_img_h, 
			img_height - center_y - simple_cut_img_h, 0))
	if padding != 0:
		simple_cut_window = int(center_x - simple_cut_img_w/2) + padding, int(center_y - simple_cut_img_h/2) + padding, simple_cut_img_w, simple_cut_img_h
		padding_img = add_padding_to_img(img, padding)
		cut_window_img = cut_window(padding_img, simple_cut_window)
	else:
		simple_cut_window = int(center_x - simple_cut_img_w/2), int(center_y - simple_cut_img_h/2), simple_cut_img_w, simple_cut_img_h
		cut_window_img = cut_window(img, simple_cut_window)
	rotate_angle = abs(abs(angle +45)-45) if angle < -45 or angle > 0 else -abs(abs(angle +45)-45)
	if False:
		print('box = ', box)
		print('boungding_window = ', bounding_window)
		print('simple_cut_window = ', simple_cut_window)
	if (is_w_bigger_than_h and w < h) or (is_w_bigger_than_h is False and w > h):
		consider90_rotate_angle = rotate_angle + 90
	else:
		consider90_rotate_angle = rotate_angle
	if False:
		print('consider90_rotate_angle = ', consider90_rotate_angle)
	rotated_cut_window_img = rotate_bound(cut_window_img, -consider90_rotate_angle)
	rotated_img_w, rotated_img_h = get_resolution(rotated_cut_window_img)
	new_center_x, new_center_y = rotated_img_w//2, rotated_img_h//2
	if False:
		print('new_center_x, new_center_y = ', new_center_x, new_center_y)
	if False:
		cv.imshow('rotated_cut_window_img', rotated_cut_window_img)
		cv.waitKey(0)

	rotated_window = int(new_center_x - new_w//2), int(new_center_y -new_h//2), int(new_w), int(new_h)
	cut_box_img = cut_window(rotated_cut_window_img, rotated_window)
	if False:
		cv.imshow('cut_box_img', cut_box_img)
		cv.waitKey(0)
	return cut_box_img
#

def contours_to_boxs(contours):
	boxs = [points_to_box(contour) for contour in contours]
	return boxs


def boxs_to_points_array(boxs):
	return [box_to_points(box) for box in boxs]


def boxs_to_imgs(img, boxs, is_w_bigger_than_h = True):
	box_imgs = [box_to_img(img, box, is_w_bigger_than_h) for box in boxs]
	return box_imgs


def convert_coord_for_point(o_point, point):
	o_x, o_y = o_point
	x, y = point
	new_point = o_x + x, o_y +y
	return new_point


def convert_minus_h_window(minus_h_window):
	left, top, w, h = minus_h_window
	window = left, top + h , w, -h
	return window


def convert_minus_h_windows(minus_h_windows):
	windows = []
	for minus_h_window in minus_h_windows:
		window = convert_minus_h_window(minus_h_window)
		windows.append(window)
	return windows


def convert_coord_for_window(o_point, window):
	o_x, o_y = o_point
	left, top, w, h = window
	new_window = o_x + left, o_y + top, w, h
	return  new_window


def add_padding_window(resolution, window, width_padding_rate, height_padding_rate):
	left, top, w, h = window
	bottomright_x, bottomright_y = left + w, top + h
	width, height = resolution
	padding_width, padding_height = int(w*width_padding_rate), int(h*height_padding_rate)
	delta_w, delta_h = padding_width - w, padding_height - h
	if left - delta_w//2 < 0:
		left = 0
	else:
		left -= delta_w//2
	if bottomright_x + delta_w//2 > width:
		bottomright_x = width
	else:
		bottomright_x += delta_w//2
	if top - delta_h//2 < 0:
		top = 0
	else:
		top -= delta_h//2
	if bottomright_y + delta_h//2 > height:
		bottomright_y = height
	else:
		bottomright_y += delta_h//2
	new_window = left, top, bottomright_x - left, bottomright_y - top
	return new_window


# draw
def draw_windows(img, windows, color=(0, 255, 0), thickness=1):
	for window in windows:
		bx, by, bw, bh = window
		cv.rectangle(img, (bx, by), (bx+bw, by+bh), color, thickness)



def draw_points_array(img, points_array, color = (0,255,255), thickness = 3):
	cv.drawContours(img, points_array, -1, color, thickness )
	# cv.polylines(img,points_array,True, color)


def get_new_color():
	global global_color_index
	global global_colors
	color_number = len(global_colors)
	color = global_colors[global_color_index%color_number]
	global_color_index += 1
	return color


def draw_boxs(img, boxs, color = (0,255,255), thickness = 3):
	for box in boxs:
		if color == -1:
			actual_color = get_new_color()
		else:
			actual_color = color
		points = box_to_points(box)
		draw_points_array(img, [points], actual_color, thickness)



def draw_points(img, points, color = (0,0,255), radius=2, thickness=-1):
	for point in points:
		cv.circle(img , point, radius, color, thickness)


def draw_np_where_points(img, y_x_array, color = (0,0,255)):
	y_array, _x_array = y_x_array
	if len(y_array) == 0:
		print('len(y_array) = ', len(y_array))
		return
	points = np_where_to_points(y_x_array)
	if False:
		print('points = ', points)
	draw_points(img, points)


def draw_information(frame, showing_window_name=None, recognitions = None, infor_dict=None, window_color=(0, 255, 0), thickness=1, full_screen=False, lang='eng'):
	standard_resolution = 960, 540
	infor_window = (20, 50, 200, 200)
	resolution = _frame_w, frame_h = get_resolution(frame)
	
	infor_l, infor_t, infor_w, infor_h = infor_window = convert_windows([infor_window], standard_resolution, resolution)[0]
	if recognitions:
		for recognition in recognitions:
			draw_windows(frame, [recognition.window], color=recognition.color, thickness=recognition.thickness)
			l, t, _w, _h = recognition.window
			if lang == 'vie':
				draw_text(frame, recognition.strg, recognition.window, color=(0, 0, 255))
			elif lang == 'eng':
				cv.putText(frame, recognition.strg, (l, t - 5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
			else:
				print('Language to draw is not accepted!')
				exit()
	if infor_dict:
		n_infor = len(infor_dict)
		distance = min(infor_h/n_infor, frame_h/10)
		for i, (key, value) in enumerate(infor_dict.items()):
			item_t = infor_t + int(i*distance)
			cv.putText(frame, str(key) + ': ' + str(value), (infor_l, item_t), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			
	if showing_window_name is None:
		return
	elif full_screen == True:
		cv.namedWindow(showing_window_name, cv.WND_PROP_FULLSCREEN)
		cv.setWindowProperty(showing_window_name,cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
	cv.imshow(showing_window_name, frame)


from PIL import Image,ImageFont, ImageDraw
def draw_text(img, text, window, fontpath="vni-full-standard/font-times-new-roman.ttf", color=(0, 255, 0)):
	n_line = len(text.split('\n'))
	l, t, w, h = window
	font_size = int(12*h/(n_line*10))
	font = ImageFont.truetype(fontpath, font_size)
	pil_img = Image.fromarray(img)
	draw = ImageDraw.Draw(pil_img)
	b,g,r,a = *color, 0
	draw.text((l, t),  text, font = font, fill = (b, g, r, a))
	img[...] = np.array(pil_img)



# video
video_extentions = ['mp4', 'avi']
img_extentions = ['jpeg', 'jpg', 'png']
def is_video_type(video_uri):
	for video_extention in video_extentions:
		if video_uri.endswith(video_extention):
			return True
	return False


def is_img_type(video_uri):
	for img_extention in img_extentions:
		if '.' + img_extention in video_uri:
			return True
	return False


def split_uri_parameters(uri):
	source = str(uri).strip()
	chunks = source.split(':')
	params = dict( s.split('=') for s in chunks[1:] )
	return params


def create_stream_video(video_uri, queueSize=8, fps=15):
	try:
		video_uri = int(video_uri)
	except:
		pass
	uri_type = None
	if isinstance(video_uri, int) or video_uri.startswith('rtsp'):
		uri_type = 'camera'
	elif video_uri.startswith('screen'):
		params = split_uri_parameters(video_uri)
		resolution = size_to_resolution(params['size'])
		
		import pyautogui
		class ScreenCapture:
			def __init__(self):
				pass
			def read(self):
				pil_img = pyautogui.screenshot()
				img = np.array(pil_img)
				img = cv.resize(img, resolution)
				if 'window' in params:
					window = text_to_window(params['window'])
					img = cut_window(img, window)
				return True, img
		screen_capture = ScreenCapture()
		return screen_capture
	if uri_type is None:
		if is_video_type(video_uri):
			uri_type = 'video'
	if uri_type is None:
		if is_img_type(video_uri):
			uri_type = 'image'
	if uri_type is None:
		print('Video uri is not accepted!')
		exit()
	if uri_type == 'image':
		import video
		camera = video.create_capture(video_uri, None)
	else:
		import videostream
		if uri_type == 'video':
			camera = videostream.QueuedStream(video_uri, fps=fps).start()
		elif uri_type == 'camera':
			camera = videostream.QueuedStream(video_uri, queueSize=queueSize).start()
		else:
			print('Something wrong when create camera!')
			exit()
	return camera



# basic img
def get_img_area(img):
	height, width, _channels = img.shape
	return height*width


def get_center(image):
	w, h = image.shape[:2]
	return (w//2, h//2)


# def resize_img(img, cvt_area):
# 	width, height = _img_revolution = img.shape[1::-1]
# 	area = width*height
# 	ratio = math.sqrt(1.0*cvt_area/area)
# 	cvt_width = int(width*ratio)
# 	cvt_height = int(height*ratio)
# 	resize_shape = (cvt_width, cvt_height)
# 	resized_img = cv.resize(img, resize_shape)
# 	return resized_img
	

def cut_window(img, window):
	if DEBUG:
		print('window = ', window)
	return img.copy()[window_to_slice(window)]


def get_resolution(img):
	resolution = _weight, _height = img.shape[1::-1]
	return resolution


def get_new_resolution(resolution, proposal_width=None, new_area=None):
	w, h = resolution
	if proposal_width:
		new_width = min(proposal_width, w)
		ratio = float(w) / new_width
	elif new_area:
		area = w*h
		ratio = math.sqrt(1.0*new_area/area)
		new_width = int(w*ratio)
	else:
		print("Use proposal_width or new_area as parameter!")
		exit()
	new_height = int(h*ratio)
	new_resolution = new_width, new_height
	return new_resolution


def resize_img(img, proposal_width=None, new_area=None):
	resolution = get_resolution(img)
	new_width, _ = new_resolution = get_new_resolution(resolution, proposal_width=proposal_width, new_area=new_area)
	resized_img = imutils.resize(img, width=new_width)
	return resized_img, new_resolution


def convert_windows(windows, source_resolution, des_resolution):
	des_w, des_h = des_resolution
	source_w, source_h = source_resolution
	w_ratio, h_ratio = float(des_w) / source_w, float(des_h)/source_h
	new_windows = []
	for window in windows:
		l, t, w, h = window
		new_window = int(l*w_ratio), int(t*h_ratio), int(w*w_ratio), int(h*h_ratio)
		new_windows.append(new_window)
	return new_windows


def check_window_in_img(resolution, window):
	img_w, img_h = resolution
	l, t, w, h = window
	r, b = l + w, t + h
	for x in [l, r]:
		if x < 0 or x > img_w:
			return False
	for y in [t, b]:
		if y < 0 or y > img_h:
			return False
	return True


def get_size_area(size):
	w, h = size
	area = w*h
	return area


def size_to_resolution(size):
	return tuple(map(lambda element: int(element), size.split('x')))


def get_resize_ratio(size,stardard_area):
	size_area = get_size_area(size)
	ratio = math.sqrt(1.0*stardard_area/size_area)
	return ratio


def get_resize_slice_step(size,stardard_area, min_step=None):
	ratio = get_resize_ratio(size,stardard_area)
	if min_step is not None:
		step = max(int(1/ratio), min_step)
	else:
		step = int(1/ratio)
	w_slice_step, h_slice_step = step, step
	return w_slice_step, h_slice_step


def create_img(img_resolution, color=0):
	width, height = img_resolution
	img = np.full((height,width,3), color, dtype=np.uint8)
	return img


def mask_to_img(mask, color=255, background_color=0):
	resolution = get_resolution(mask)
	img = create_img(resolution, background_color)
	img[mask==1] = color
	return img


def get_name_file_without_ext_from_path(path):
	name_file = path[path.rfind('/')+1: path.rfind('.')]
	return name_file


def box_transform_by_resolution(box, des_resolution, source_resolution):
	srs_width, src_height = source_resolution
	dst_width, dst_height = des_resolution
	width_ratio, height_ratio=_resolution_ratio = 1.0*dst_width/srs_width, 1.0*dst_height/src_height
	(center_x, center_y), (a, b), angle = box
	new_box = (center_x*width_ratio, center_y*height_ratio), (a*width_ratio, b*height_ratio), angle
	return new_box


def boxs_transform_by_resolution(boxs, des_resolution, source_resolution):
	srs_width, src_height = source_resolution
	dst_width, dst_height = des_resolution
	width_ratio, height_ratio=_resolution_ratio = 1.0*dst_width/srs_width, 1.0*dst_height/src_height
	
	new_boxs = []
	for box in boxs:
		(center_x, center_y), (a, b), angle = box
		new_box = (center_x*width_ratio, center_y*height_ratio), (a*width_ratio, b*height_ratio), angle
		new_boxs.append(new_box)
	return new_boxs


def get_window_area(window):
	_x, _y, w, h = window
	return w*h


def get_box_area(box):
	(_center_x, _center_y), (w, h), _angle = box
	return int(w*h)


def get_diff_angle_between_two_boxs(box1, box2):
	(_, _), (_, _), angle1 = box1
	(_, _), (_, _), angle2 = box2
	return math.sqrt(angle1 -angle2)


def write_imgs(folder, img_name, imgs):
	try:
		os.stat(folder)
	except:
		os.mkdir(folder)
	for i, img in enumerate(imgs):
		cv.imwrite(folder + '/' + img_name + '_' + str(i) + '.jpg', img)


def write_img(folder, file_name,  img):
	try:
		os.stat(folder)
	except:
		os.mkdir(folder)
	cv.imwrite(folder +'/'+file_name, img)


def plt_imshow(name_window, img):
	# cv.imshow(name_window, img)
	plt.title(name_window)
	plt.imshow(img)
	# plt.show()


def add_bouding(img, thickness = 1):
	img = img.copy()
	width, height = get_resolution(img)
	window = thickness,thickness, width-thickness, height-thickness
	draw_windows(img, [window], (0,0,0), thickness)
	return img


def morphloEx(image, shape = cv.MORPH_ELLIPSE ,shape_size = (3,3), op = cv.MORPH_ERODE, iterations = 1):
	image = image.copy()
	structure = cv.getStructuringElement(shape, shape_size)
	new_img = cv.morphologyEx(image, op, structure, iterations)
	return new_img


# advance img
def compare_two_masks(mask1, mask2, proposal_size=20):
	size =  min(min(mask1.shape + mask2.shape), proposal_size)
	standard_resolution = size, size
	resized_mask1 = cv.resize(mask1, standard_resolution)
	resized_mask2 = cv.resize(mask2, standard_resolution)
	similar_points = np_boolean_array_to_points(resized_mask1==resized_mask2)
	similar_value = (2*len(similar_points)+1)/(np.count_nonzero(resized_mask1)+np.count_nonzero(resized_mask2)+1)
	return similar_value


def grow_region(img, seed_pt, threshold=30):
	w, h = get_resolution(img)
	lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
	low = high = threshold 
	floodFill_mask = np.zeros((h+2, w+2), np.uint8)
	region_window = floodFill_return_window(lab_img, floodFill_mask, seed_pt,low, high)
	region_mask = floodFill_mask[1:-1, 1:-1][window_to_slice(region_window)]
	return region_window, region_mask


# rotate
def rotate_img(img, angle):
	rows,cols = img.shape[:-1]
	M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv.warpAffine(img,M,(cols,rows))
	return dst


def find_rotated_angle_from_rect_boxs(rect_boxs):
	angles = [box[-1] for box in rect_boxs]
	differrence = [-abs(abs(angle +45)-45) if angle < -45 or angle > 0 else abs(abs(angle +45)-45) for angle in angles]
	diff = statistics.median(differrence)
	return -diff


def rotate_bound(img, angle):
	image = img.copy()
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos)) 
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW // 2) - cX
	M[1, 2] += (nH // 2) - cY
	# perform the actual rotation and return the image
	return cv.warpAffine(image, M, (nW, nH))


def getRotationMatrix2D(image, angle):
	w, h = get_resolution(image)
	(cX, cY) = (w // 2, h // 2)
	M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
	return M


def convert_points_by_rotation_matrix(points, matrix):
	
	if DEBUG:
		print('points = ',points)
	ones = np.ones(shape=(len(points), 1))
	points_ones = np.hstack([points, ones])
	# transform points
	new_points = matrix.dot(points_ones.T).T
	if DEBUG:
		print('new_points = ',new_points)
	new_points = new_points.astype(np.int64)

	if DEBUG:
		print('new_points after change dtype to int = ',new_points)
	return new_points


def window_overlaping_area(window1, window2):
	left1, top1, w1, h1 = window1
	left2, top2, w2, h2 = window2
	bottomright_x1, bottomright_y1 = left1 + w1, top1 + h1
	bottomright_x2, bottomright_y2 = left2 + w2, top2 + h2
	x_overlap = max(0, min(bottomright_x1, bottomright_x2)-max(left1, left2))
	y_overlap = max(0, min(bottomright_y1, bottomright_y2)-max(top1, top2))
	overlap_area = x_overlap * y_overlap
	return overlap_area


def cal_box_overlaping_area_ratio(box1, box2):
	two_boxs_points = np.concatenate([box_to_points(box1), box_to_points(box2)])
	two_boxs_window = points_to_window(two_boxs_points)
	if False:
		print('two_boxs_window = ', two_boxs_window)
	topleft, bottomright = _two_points = window_to_two_points(two_boxs_window)
	if False:
		print('topleft, bottomright = ', topleft, bottomright)
	img_resolution = tuple(bottomright)
	if False:
		print('img_resolution = ', img_resolution)
	if False:
		print('box1, box2 = ', box1, box2)
	img = create_img(img_resolution, is_value_is_zero= False)
	(center_x1, center_y1), (a1, b1), angle1 = box1
	(center_x2, center_y2), (a2, b2), angle2 = box2

	new_center_x1, new_center_y1 = center_x1 - topleft[0], center_y1 - topleft[1]
	new_center_x2, new_center_y2 = center_x2 - topleft[0], center_y2 - topleft[1]
	box1 = (new_center_x1, new_center_y1), (a1, b1), angle1
	box2 = (new_center_x2, new_center_y2), (a2, b2), angle2
	if False:
		print('new box1, box2 = ', box1, box2)
	img1 = img.copy()
	img2 = img.copy()
	(_center_x1, _center_y1), (_w1, _h1), _angle1 = box1
	(_center_x2, _center_y2), (_w2, _h2), _angle2 = box2
	points1, points2 = box_to_points(box1), box_to_points(box2)
	img1 = cv.fillConvexPoly(img1, points1, (0,0,0))
	img2 = cv.fillConvexPoly(img2, points2, (0,0,0))
	mask1 = img1 == (0, 0, 0)
	mask2 = img2 == (0, 0, 0)
	and_mask = mask1 & mask2
	count = np.sum(and_mask)
	rate = 2.0*count/(np.sum(mask1) + np.sum(mask2))
	return rate


def is_overlap(window1, window2):
	overlap_area = window_overlaping_area(window1, window2)
	if overlap_area > 0:
		return True
	return False


def check_rotate_90(rect_boxs, is_w_bigger_than_h = True):
	count = 0
	for box in rect_boxs:
		window = box_to_window(box)
		_, _, w, h = window
		if DEBUG:
			print('window = ', window)
		if w > h:
			count += 1
	threadhole = len(rect_boxs)//2
	if DEBUG:
		print('count = ', count)
		print('threadhole = ', threadhole)

	if count > threadhole:
		return False
	return True


# numpy
def index_to_numpy_point(index, shape):
	_l_dim1, l_dim2 = shape
	dim1 = index//l_dim2
	dim2 = index%l_dim2
	point = (dim1, dim2)
	return point


def numpy_point_to_point(numpy_point):
	dim1, dim2 =  numpy_point 
	point = _x,_y = dim2, dim1
	return point


def point_to_numpy_point(point):
	dim2, dim1 = point
	np_point = dim1, dim2
	return np_point


def index_to_point(index, shape):
	numpy_point = index_to_numpy_point(index, shape)
	point = numpy_point_to_point(numpy_point)
	return point


def window_to_slice(window, slice_step=None):
	left, top, w, h = window
	if slice_step is None:
		window_slice = slice(top, top+h), slice(left,left+w)
	else:
		w_slice_step, h_slice_step = slice_step
		window_slice = slice(top, top+h, h_slice_step), slice(left,left+w, w_slice_step)
	return window_slice


def np_where_to_points(y_x_array):
	points = []
	for y,x in zip(y_x_array[0], y_x_array[1]):
		points.append((x,y))
	return points


# def np_where_to_points(x_y_array):
# 	points = []
# 	for x,y in zip(x_y_array[0], x_y_array[1]):
# 		points.append((x,y))
# 	return points


def np_boolean_array_to_points(np_boolean_array):
	np_points = list(zip(*np.where(np_boolean_array)))
	points = list(map(numpy_point_to_point, np_points))
	return points


def convert_coord_np_where_points(y_x_array, o_point):
	ox, oy = o_point
	y_array, x_array = y_x_array
	converted_y_x_array = _converted_y_array, _converted_x_array = y_array + oy, x_array + ox
	return converted_y_x_array


def random_pick_return_index(probabilities):
	x = random.uniform(0, 1)
	cumulative_probability = 0.0
	for i, item_probability in enumerate(probabilities):
		cumulative_probability += item_probability
		if x < cumulative_probability: break
	return i


def probability_random_point(probabilities, mask=None, reverse_matrix=False):
	if np.all(mask==0):
		return None
	if reverse_matrix:
		probabilities = np.max(probabilities) + 1 - probabilities 
		probabilities = probabilities/np.sum(probabilities)
	if mask is None:
		w, h = get_resolution(probabilities)
		mask = np.ones(h, w)
	np_where_points = np.where(mask==1)
	points = np_where_to_points(np_where_points)
	if not points:
		print('No point satify!')
		return None
	point_probabilities = []
	for x, y in points:
		probability = probabilities[y,x]
		point_probabilities.append(probability)
	randomed_index = random_pick_return_index(point_probabilities)
	randomed_point = points[randomed_index]
	return randomed_point


def pb_random_point_from_count_array(count):
	if DEBUG:
		print('count = ', count)
		print('count.shape = ', count.shape)
	origin_shape = count.shape
	reshape_count = count.copy().reshape(-1)
	# count.shape = (len_dim1*len_dim2,)
	# if DEBUG:
	# 	print('origin_shape = ', origin_shape)
	# 	print('self.count = ', count)
	probabilities = max(reshape_count) - reshape_count 
	probabilities = probabilities/np.sum(probabilities)	
	if DEBUG:
		print('probabilities = ', probabilities)
	# count.shape = origin_shape
	index = random_pick_return_index(probabilities)
	point = index_to_point(index, origin_shape)
	return point


def distance_between_two_points(a, b):
	return math.sqrt(sum((a - b)**2 for a, b in zip(a, b)))


def distance_between_two_windows(window1, window2):
	_unit1 = center1, (w1, h1)= window_to_unit(window1)
	_unit2 = center2, (w2, h2)= window_to_unit(window2)
	center_distance = distance_between_two_points(center1, center2)
	simple_distance = (int(center_distance) + abs(w1//2-w2//2) + abs(h1//2-h2//2))/3
	distance = 2*simple_distance/(math.sqrt(get_window_area(window1))+math.sqrt(get_window_area(window2)))
	return distance


def set_value_for_array(np_array, window, mask, value, operation=None):
	focus_np_array = np_array[window_to_slice(window)]
	if not operation:
		focus_np_array[mask==1] = value
		return
	elif operation == '+':
		focus_np_array[mask==1] += value
	elif operation == '-':
		focus_np_array[mask==1] -= value
	elif operation == '*':
		focus_np_array[mask==1] *= value
	elif operation == '/':
		focus_np_array[mask==1] /= value
	else:
		print('Operation is not accepted!')
		exit()



def arrays_to_dict_array(key1, values1, key2, values2):
	array = []
	for value1, value2 in zip(values1, values2):
		dictionary = {}
		dictionary[key1] = value1
		dictionary[key2] = value2
		array.append(dictionary)
	return array


def window_text_dicts_to_font_size(text, letter_windows, standard_font):
	h_font_sizes = []
	for letter, letter_window in zip(text, letter_windows):
		_left, _top, _w, h = letter_window
		(_standard_font_width, standard_font_baseline), (_offset_x_standard_font, _offset_y_standard_font) = standard_font.font.getsize(letter)
		if standard_font_baseline == 0:
			continue
		h_font_size = int(12.0/(standard_font_baseline)*h)
		h_font_sizes.append(h_font_size)
	if not h_font_sizes:
		return 1
	median_h_font_size = int(np.median(h_font_sizes))
	return median_h_font_size


# floodFill
def window_to_floodFill_mask(window):
	left, top, w, h = window
	floodFill_mask = np.zeros((h+2, w+2), np.uint8)
	return floodFill_mask


def none_margin_mask_to_mask(none_margin_mask_to_mask):
	w, h = get_resolution(none_margin_mask_to_mask)
	floodFill_mask = np.empty((h+2, w+2), np.uint8)
	floodFill_mask[1:-1,1:-1][:] = none_margin_mask_to_mask
	return floodFill_mask


def floodFill_return_window(img, floodFill_mask, seed_pt, low, high, color = (255, 255, 255), flags = cv.FLOODFILL_FIXED_RANGE):
	hsv = cv.cvtColor(img, cv.COLOR_BGR2LAB)
	cv.floodFill(hsv, floodFill_mask, seed_pt, (255, 255, 255), (low,low,low), (low,low,low), flags)
	dim2, dim1 = seed_pt
	floodFill_mask[1:-1, 1:-1][dim1, dim2] = 1
	none_margins_mask = floodFill_mask[1:-1, 1:-1]
	pixelpoints = cv.findNonZero(none_margins_mask)
	window = points_to_window(pixelpoints)
	return window


# ocr
import pytesseract
def image_to_string(img, lang='vie'):
	img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
	if lang == 'eng':
		# config = ("-l eng ")
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(img, config=config)
	elif lang == 'vie':
		# config = ("-l vie --oem 3 --psm 6")
		config = ("-l vie --oem 3 --psm 7")
		text = pytesseract.image_to_string(img, config=config)	
	else:
		print('Language is not accepted!')
		exit()
	return text


# important function
def copy(ob):
	import copy
	return copy.deepcopy(ob)

# 
def check_meaningful_letters(letters):
	for letter in letters:
		if not letter.isspace() and letter != '~' and letter != '':
			return True
	return False


def center_of_4points(points):
	(x1,y1), (x2,y2), (x3,y3), (x4, y4) = points
	xi = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
	yi = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
	return xi, yi

	
def calculate_angle_vector_and_vertical_vector(vector):
	x, y = vector
	vertical_vector = np.array([0, 1])
	# vertical_vector = np.array([1, 0])

	vector = np.array(vector)
	u, v = vertical_vector, vector
	c = dot(u,v)/norm(u)/norm(v) 
	angle = arccos(clip(c, -1, 1))
	if x < 0:
		angle = 2*math.pi - angle 
	if angle == 2*math.pi:
		angle = 0
	return angle



def string_to_int(number):
	return int(number)


def text_to_window(text):
	if text.startswith('('):
		text = text[1:]
	if text.endswith(')'):
		text = text[:-1]
	string_elements = [e.strip() for e in text.split(',')]
	window = tuple(map(string_to_int, string_elements))
	return window 


def window_to_text(window):
	 return ', '.join(map(str, window))


####

def load_config(path):
	config = {}
	fn = open(path, 'r')
	for line in fn.readlines():
		key, value = line.split('=') 
		if is_int(value):
			true_type_value = int(value)
		elif is_float(value):
			true_type_value = float(value)
		else:
			if value.endswith('\n'):
				true_type_value = value[:-1]
			else:	
				true_type_value = value
		config[key] = true_type_value
	return config


def dump_config(config, path):
	fn = open(path, 'w')
	for key, value in config.items():
		print(value)
		fn.write(str(key)+'='+str(value)+'\n')


def is_float(s):
	try:
		float(s)
		return True
	except ValueError:
		return False


def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False
