Assignment #3: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

Image segmentation is the division of an image into regions or categories, which correspond to different objects or parts of objects. 
After segmentation, the output is a region or a structure that collectively covers the entire image.
These regions have similar characteristics including colors, texture, or intensity. 
Image segmentation is important because it can enhance the analysis of images with more granularity. 
It extracts the objects of interest, for further processing such as description or recognition.
Image segmentation works by using encoders and decoders. Encoders take in input, which is a raw image, it then extracts features, and finally, decoders generate
an output which is an image with segments.
![image](https://user-images.githubusercontent.com/11517432/175871527-95b18995-5695-4775-aae0-51e3c34d1e29.png)

The most common architectures in image segmentation are U-Net, FastFCN, Deeplab, Mask R-CNN, etc.
Your job is to use U-Net architecture with convolutional layers to build a model for the segmentation task.
