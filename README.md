# AI-Cheque-Processing-System

## Summary

In this modern era, many people still rely on cheques to transfer money and as a result, millions of cheques are processed daily using manual and template based approaches.
However, these methods are fixed rule based approaches and lead to a lot of inefficiency even if minor errors occur in the cheque. 

This project aims to propose a method to use AI as a means to dynamically process cheques, irrespective of what format they are in, and hence improve the efficiency as well as the accuracy of the system. This also enables small corporations to directly use their scanners to process cheques, and not buy expensive cheque reading machines that maybe limited to a particular format. 

##Working

This project uses segmentation and OCR to extract data from the cheque and present them to the user. 

1. Photo of the cheque is uploaded to the system. 
2. The photo is passed through a segmentation network known as U-Net to predict the areas in which a certain type of data can be found. 
3. These areas are then extracted and smoothened out to get a definie ROI for the data. 
4. The ROIs are passed through OCR in order to get the data out of the ROIs

OCR for the MICR text is custom trained on a MICR based font data using several variations of 1 million MICR code ROIs of cheques. As for the other fields, Google Vision OCR has been used to provide the highest accuracy even for the regions with accidental smudges or in general for a image of less quality. 

U-Net is trained on self collected data of around 1000 check iamges, which was then manually labelled through the labelme tool. 

## Getting Started
```python 
!cd Solution
python main.py --image image_name
```
