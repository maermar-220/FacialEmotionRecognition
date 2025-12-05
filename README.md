## Demo Description

This notebook demonstrates a simple end-to-end pipeline for facial expression recognition from a single image. The workflow is organized into three main cells:

1. **Setup & Model Definitions**   
   - Defines helper functions for RetinaFace (priors, decoding, NMS, face alignment, preprocessing)  
   - Defines the `RAFStage1Plus` expression recognition model (ResNet18-based) and image transforms

2. **Face Detection and Alignment (RetinaFace ONNX)**  
   - Loads the input image `test.jpeg`  
   - Runs face detection and landmark prediction using the RetinaFace ONNX model  
   - Decodes bounding boxes and landmarks, applies NMS, and aligns each detected face to 224×224  
   - Saves:
     - The original image with detection boxes and landmarks  
     - Each aligned face crop into a separate folder  
   - Stores detection results in a global variable for the next step

3. **Facial Expression Recognition (PyTorch)**  
   - Loads the pretrained expression recognition model from  
     `/content/raf_stage1plus_best2.pth`  
   - Applies the model to each aligned face crop  
   - Outputs the predicted emotion label and confidence for each face, and visualizes them in the notebook
