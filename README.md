# Cloud-Segmentation-UNET
This repository contains the first model which I tried on cloud dataset from sentinel satellite for cloud segmentation. The model was UNET trained for 15 epochs and the loss function used was BinaryCrossEntropy with Logits. For, validation IoU was used. There were 7189 images in training dataset and 4811 images on validataion dataset. Each Image had 4 channels - Red, Green, Blue and Infrared.
Accuracy of 81.39% could be achieved on the validation set using this model.

-- Some Snaps of Result --

First Image is a channel of the given image, 2nd image is true label, 3rd image is the prediction.

![image](https://user-images.githubusercontent.com/79797859/151528644-f5a77412-f8f8-4283-8556-41b1ffb28e7c.png)
![image](https://user-images.githubusercontent.com/79797859/151528805-829c96dd-b3e4-4761-9c0d-6eda9b5a746a.png)

