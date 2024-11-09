# MediGen: AI-Based Medical Report Generator

This project classifies X-ray images into specific disease categories and generates a well-documented medical report based on the predicted disease. The model uses **ResNet50**, a deep convolutional neural network pretrained on ImageNet, for feature extraction and disease classification. The project then generates a textual medical report summarizing the disease detected.

## Methodology

1. **Load Metadata and Image Paths**  
   The project begins by loading a CSV file that contains metadata with image filenames and their corresponding disease labels. The images are then mapped to their full file paths.

2. **Preprocess Images**  
   - Images are read and decoded in PNG format.
   - Each image is resized to 224x224 pixels, the required input size for **ResNet50**.
   - Pixel values are normalized to the range [0, 1].

3. **Prepare Dataset**  
   - Pair images with their corresponding disease labels (e.g., "Pneumonia", "COVID-19", "Tuberculosis", etc.).
   - Prepare a dataset that can be used for training the model.
   - Use TensorFlow's `tf.data` pipeline for efficient batching, shuffling, and prefetching.

4. **Model Architecture (ResNet50)**  
   - **ResNet50** is used as the backbone for feature extraction, utilizing pretrained weights from ImageNet.
   - The output from ResNet50 is passed through a Global Average Pooling layer and a Dense layer with **softmax activation** to predict one of several possible diseases.

5. **Train the Model**  
   The model is compiled using the **Adam** optimizer and trained with **categorical cross-entropy** loss, which is appropriate for multi-class classification tasks.

6. **Evaluate the Model**  
   The model is evaluated on a validation set to assess its accuracy in classifying the X-ray images into the correct disease categories.

7. **Generate Medical Report**  
   After the model makes predictions, a textual report is generated summarizing the detected disease.

## Requirements

- Python 3.x
- TensorFlow >= 2.0
- NumPy
- Pandas
- OpenCV (for image processing)
- scikit-learn (for model evaluation)
