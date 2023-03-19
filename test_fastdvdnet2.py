#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import numpy as np
import os, gc
import argparse
import time
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger, open_image

NUM_IN_FR_EXT = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

def save_out_seq(cnt, seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('{}').format(cnt) + fext)
			cnt+=1
		else:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def test_fastdvdnet(**args):
	"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_file": path to model
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
			"gray": if True, perform denoising of grayscale images instead of RGB
			"video":if True, path is a video
			"batch_size":set the batch size depending on available gpu

	"""

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# Create models
	if args['debug']:
		print('Loading models ...')
	model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
	if args['debug']:
		print("Model created")
	# Load saved weights
	state_temp_dict = torch.load(args['model_file'], map_location='cpu')
	if args['cuda']:
		device_ids = [0]
		model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
	else:
		# CPU mode: remove the DataParallel wrapper
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_temp.load_state_dict(state_temp_dict)
	if args['debug']:
		print("Loaded weights")
	# Sets the model in evaluation mode (e.g. it removes BN)
	model_temp.eval()
	print(args['test_path'])
	if not os.path.exists(args['test_path']):
		print("Video does not exist")
	else:
		print("Found video")
	frame_skip = args['frame_sampling']
	cap = cv2.VideoCapture(args['test_path'])
	frame_rate = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count/ frame_rate
	print("Frame count = ", frame_count)
	print("Video Duration = ", duration)
	print("Frame Rate = ", frame_rate)
	count = 0
	fr_cnt = 0
	frames = []
	total_frames = 0
	# while cap.isOpened():
	for count in tqdm(range(frame_count)):
		ret, frame = cap.read()
		
		# if not ret:
		# 	break
		
		# count +=1
		if not count % frame_skip == 0:
			continue

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		frame = cv2.flip(frame, 0)
		frame = frame.transpose(2, 1, 0)
		# Save the frame as an image
		frames.append(frame)
		fr_cnt+=1
		# print("Frame: ", count, "/", frame_count, end = '\r')
		if not fr_cnt % args['batch_size'] == 0:
			continue
		
		with torch.no_grad():
			if args['debug']:
				print("Process data")
			# process data
			seq_list = []			
			for frame in frames:
				img, _ , _ = open_image(frame,\
											gray_mode=args['gray'],\
											expand_if_needed=False,\
											expand_axis0=False)
				seq_list.append(img)
			seq = np.stack(seq_list, axis=0)
			seq = torch.from_numpy(seq).to(device)
			if args['debug']:
				print("Add noise")
			# Add noise
			# noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
			seqn = seq # seq + noise
			noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)
			if args['debug']:
				print("Denoising")
			denframes = denoise_seq_fastdvdnet(seq=seqn,\
											noise_std=noisestd,\
											temp_psz=NUM_IN_FR_EXT,\
											model_temporal=model_temp)
			# print("Saving image")
			if not args['dont_save_results']:
				# Save sequence
				save_out_seq(fr_cnt-args['batch_size'], seqn, denframes, args['save_path'], \
							int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])
			del denframes
		gc.collect()
		torch.cuda.empty_cache()
		frames = []
	cap.release()
	img = cv2.imread(f'results/0.png')
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	height, width, layers = img.shape
	size = (width,height)
	print("New farme rate: ", frame_rate/frame_skip)
	out = cv2.VideoWriter(f"{args['save_path']}/output.avi",cv2.VideoWriter_fourcc(*'MJPG'), frame_rate/frame_skip, size)
	if (out.isOpened() == False):
		print("Error reading video file")
	for i in tqdm(range(fr_cnt//args['batch_size'] * args['batch_size'])):
        # writing to a image array
		img = cv2.imread(f'results/{i}.png')
		height, width, layers = img.shape
		size = (width,height)
		out.write(img)
	
	out.release()
	cv2.destroyAllWindows()
 
	file_size = os.path.getsize('results/output.avi')
	print("File Size is :", file_size/(1024*1024), "megabytes")

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,\
						default="./model.pth", \
						help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="data_set/videoplayback.mp4", \
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=4, \
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=30, help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")

	parser.add_argument("--save_path", type=str, default='./results', \
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',\
						help='perform denoising of grayscale images instead of RGB')
	parser.add_argument("--video", action='store_true', help="Path is not a video")
	parser.add_argument("--batch_size", type=float, default=4, help="set the batch size depending on available gpu")
	parser.add_argument("--frame_sampling", type=float, default=1, help="1/frame_sampling of the frame samples will be taken")
	parser.add_argument("--debug", action='store_true', help="Printing will occur")

	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()


	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_fastdvdnet(**vars(argspar))
