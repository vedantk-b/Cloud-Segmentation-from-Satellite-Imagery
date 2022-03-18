# Cloud-Segmentation from Sentinel 10 Satellite Imagery

## Overview 

- Satellite imagery is critical for a wide variety of applications from disaster management and recovery, to agriculture, to military intelligence. Clouds present a major obstacle for all of these use cases, and usually have to be identified and removed from a dataset before satellite imagery can be used. Improving methods of identifying clouds can unlock the potential of an unlimited range of satellite imagery use cases, enabling faster, more efficient, and more accurate image-based research.
- Our goal was to detect cloud cover in satellite imagery to remove cloud interference. The challenge used publicly available satellite data from the Sentinel-2 mission, which captures wide-swath, high-resolution, multi-spectral imaging. For each tile, data is separated into different bands of light across the full visible spectrum, near-infrared, and infrared light.
- To accompllish the task, we tried several segmentation models like UNET with Resnet18 encoder, DeepLabV3 with Resnet50 encoder, DeepLabV3+ with Resnet101 endoer and Pix2Pix GAN.
- First We trained on 4 channel images and then, to improve the accuracy, we trained on 12 channel images.
- For training, SGD and Adam optimizers were used with different learning rates ranging from 2e-3 to 3e-7.
- The Loss function used was BinaryCrossEntropy Loss, with and without logits, depending whether the model had softmax layer at the end or not.
- All the models were made using Pytorch on Jupyter Notebook on the Microsoft Planetary Computer.
- This was a code competition and a code had to be submitted which would retrive data from the computer and then do the prediciton on that and save the predictions in a given directory.
- The submission file for 4 channels was simple because all the 4 channels were present in the remote computer.
- The 12 channel submission file downloaded the other channels for a particular image on the remote server, predicted and saved and did the same for each image.


-- Some Snaps of Result --

First Image is a channel of the given image, 2nd image is true label, 3rd image is the prediction.

![image](https://user-images.githubusercontent.com/79797859/151528644-f5a77412-f8f8-4283-8556-41b1ffb28e7c.png)
![image](https://user-images.githubusercontent.com/79797859/151528805-829c96dd-b3e4-4761-9c0d-6eda9b5a746a.png)

