# Cloud-Segmentation on satellite data from the [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) mission.

## Problem Description
To obtain adequate analytical results from multi-spectral satellite imagery, it is essential to precisely detect clouds and mask them out from any Earth surface as they obscure important ground-level features in satellite images, complicating their use in wide variety of applications from disaster management and recovery, to agriculture, to military intelligence. Thus, Improving methods of identifying clouds can unlock the potential of an unlimited range of satellite imagery use cases, enabling faster, more efficient, and more accurate image-based research.

## Dataset
- The challenge used publicly available satellite data from the Sentinel-2 mission, which captures wide-swath, high-resolution, multi-spectral imaging. There are four images associated with each chip. Each image within a chip captures light from a different range of wavelengths, or "band".

| Band | Description         | Center wavelength |
|------|:--------------------|-------------------|
| B02  | Blue visible light  | 497 nm            |
| B03  | Green visible light | 560 nm            |
| B04  | Red visible light   | 665 nm            |
| B08  | Near infrared light | 835 nm            |

- For more details visit: https://www.drivendata.org/competitions/83/cloud-cover/page/398/#images

## Getting Started
- Creating new conda environment:
  ```shell
    conda create -n cloudncloud -f environment.yml
    conda activate cloudncloud
    pip install segmentation_models_pytorch
    ```

- Set number of epochs, batch size, optimizer, loss function, model, transformation to be applied on data by
changing them in ```config.py```

- To use Unet with inceptionv4 as backbone
    ```py
  model = smp.Unet(
                 encoder_name="inceptionv4",
                 in_channels=4,
                 classes=2
                )
  ```

- To use DeepLabV3 with resnet101 as backbone
    ```py
  model = smp.DeepLabV3(
               encoder_name="resnet101",
               in_channels=4,
               classes=2
            )
    ```
- Training and validation loops can be customised by editing

## Results

| Model Name                               | Public mIoU Score | Private mIoU Score |
|------------------------------------------|-------------------|--------------------|
| DeepLabV3Plus with ResNet101 as backbone | 0.8805            | 0.8775             |
| Unet with InceptionV4 as backbone        | 0.8776            | 0.8749             |
| DeepLabV3 with ResNet101 as backbone     | 0.8299            | 0.8340             |

The best accuracy was achieved with __DeepLabV3Plus with ResNet101 as backbone.__

1st Image is a channel of the satellite image, 2nd image is true label, 3rd image is the prediction.

![image](https://user-images.githubusercontent.com/79797859/151528644-f5a77412-f8f8-4283-8556-41b1ffb28e7c.png)
![image](https://user-images.githubusercontent.com/79797859/151528805-829c96dd-b3e4-4761-9c0d-6eda9b5a746a.png)

## People

| Vidit Agarwal                                             | Vedant Kaushik                                            | Utkarsh Pandey                                            |
|-----------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| ![](https://avatars.githubusercontent.com/u/72168180?v=4) | ![](https://avatars.githubusercontent.com/u/79797859?v=4) | ![](https://avatars.githubusercontent.com/u/78856460?v=4) |
| [https://github.com/Viditagarwal7479](https://github.com/Viditagarwal7479)                                                      | [https://github.com/vedantk-b](https://github.com/vedantk-b)                                                      | [https://github.com/Kratos-is-here](https://github.com/Kratos-is-here)                     |
