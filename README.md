# New-Augmentation-Method
New method for image dataset augmentation 
Random image background change using deep mediapip library

Methods:
data-augmentation method based on semantic segmentation. The MobileNetV3 model was used to get an image semantic segmentation mask, which was used to get a binary mask, which in turn was used to replace the image background by using conditional subtraction with randomly selected background images. Finally, randomly pixel-based color augmentation was added to the resulting image"
Resulted images


<img width="482" alt="image" src="https://github.com/user-attachments/assets/a554bd92-2ff4-4e5a-99db-d45f3bf20c22" />
