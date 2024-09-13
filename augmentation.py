import numpy as np
import albumentations as A
import cv2

def flip_up_down(image, ground_truth):
    # print('Flipping Data up and down')
    # print(f'recive : image : {image.shape} {image.dtype} ::  ground_truth { ground_truth.shape} { ground_truth.dtype} ')
    transform = A.Compose([A.VerticalFlip(p=1.0)])
    transformed_image = transform(image=image)['image']
    transformed_gt = transform(image=ground_truth)['image']
    # print(f'data Typye check:  \n transformed_image  -> shape :  {transformed_image.shape} dataytpe:  {transformed_image.dtype}:: \n transformed_gt  :: shape  {transformed_gt.shape}   dtype  {transformed_gt.dtype}  ')
    # print()
    return transformed_image, transformed_gt, 

def flip_left_right(image, ground_truth):
    # print('Shifting data left to right on x axis')
    # print(f'recive : image : {image.shape} {image.dtype} ::  ground_truth { ground_truth.shape} { ground_truth.dtype} ')
    transform = A.Compose([A.HorizontalFlip(p=1.0)])
    transformed_image = transform(image=image)['image']
    transformed_gt = transform(image=ground_truth)['image']
    # print(f'data Typye check:  \n transformed_image  -> shape :  {transformed_image.shape} dataytpe:  {transformed_image.dtype}:: \n transformed_gt  :: shape  {transformed_gt.shape}   dtype  {transformed_gt.dtype}  ')
    # print()
    return transformed_image, transformed_gt

def transpose(image, ground_truth):
    # print('Applying Transpose')
    # print(f'recive : image : {image.shape} {image.dtype} ::  ground_truth { ground_truth.shape} { ground_truth.dtype} ')
    transform = A.Compose([A.Transpose(p=1.0)])
    transformed_image = transform(image=image)['image']
    transformed_gt = transform(image=ground_truth)['image']
    # print(f'data Typye check:  \n transformed_image  -> shape :  {transformed_image.shape} dataytpe:  {transformed_image.dtype}:: \n transformed_gt  :: shape  {transformed_gt.shape}   dtype  {transformed_gt.dtype}  ')
    # print()
    return transformed_image, transformed_gt

def rotate_90(image, ground_truth):
    # print('Rotating 90 degrees')
    # print(f'recive : image : {image.shape} {image.dtype} ::  ground_truth { ground_truth.shape} { ground_truth.dtype} ')
    transform = A.Compose([A.Rotate(limit=(90, 90), p=1.0)])
    transformed_image = transform(image=image)['image']
    transformed_gt = transform(image=ground_truth)['image']
    # print(f'data Typye check:  \n transformed_image  -> shape :  {transformed_image.shape} dataytpe:  {transformed_image.dtype}:: \n transformed_gt  :: shape  {transformed_gt.shape}   dtype  {transformed_gt.dtype}  ')
    # print()
    return transformed_image, transformed_gt

def rotate_180(image, ground_truth):
    # print('Rotating 180 degrees')
    # print(f'recive : image : {image.shape} {image.dtype} ::  ground_truth { ground_truth.shape} { ground_truth.dtype} ')
    transform = A.Compose([A.Rotate(limit=(180, 180), p=1.0)])
    transformed_image = transform(image=image)['image']
    transformed_gt = transform(image=ground_truth)['image']
    # print(f'data Typye check:  \n transformed_image  -> shape :  {transformed_image.shape} dataytpe:  {transformed_image.dtype}:: \n transformed_gt  :: shape  {transformed_gt.shape}   dtype  {transformed_gt.dtype}  ')
    # print()
    return transformed_image, transformed_gt

def rotate_270(image, ground_truth):
    # print('Rotating 270 degrees')
    # print(f'recive : image : {image.shape} {image.dtype} ::  ground_truth { ground_truth.shape} { ground_truth.dtype} ')
    transform = A.Compose([A.Rotate(limit=(270, 270), p=1.0)])
    transformed_image = transform(image=image)['image']
    transformed_gt = transform(image=ground_truth)['image']
    # print(f'data Typye check:  \n transformed_image  -> shape :  {transformed_image.shape} dataytpe:  {transformed_image.dtype}:: \n transformed_gt  :: shape  {transformed_gt.shape}   dtype  {transformed_gt.dtype}  ')
    # print()
    return transformed_image, transformed_gt