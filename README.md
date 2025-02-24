# 🩺 Smart Chest Disease Detection using Deep Learning  

## Overview  
This project leverages deep learning techniques to automate the detection and classification of chest diseases from X-ray images. Using the **Faster R-CNN** architecture, the model aids healthcare professionals in diagnosing conditions such as pneumonia, tuberculosis, and lung cancer. With the **NIH ChestX-ray14** dataset, this solution delivers robust and accurate results to enhance diagnostic processes.

---

## Objectives  
- Develop an AI-powered model for detecting thoracic diseases using Faster R-CNN.  
- Optimize model performance through metrics such as accuracy, precision, recall, and F1-score.  
- Utilize techniques like data augmentation and transfer learning to enhance model accuracy.  
- Produce interpretable outputs by generating bounding boxes around areas of interest.  

---

## Technologies Used  
- **Frameworks:** PyTorch, Detectron2  
- **Model:** Faster R-CNN with ResNet-50 backbone  
- **Visualization Tools:** Matplotlib, Seaborn  
- **Utilities:** OpenCV, Google Colab  
- **Dataset:** NIH ChestX-ray14  

---

## Dataset  
The **NIH ChestX-ray14** dataset includes over **112,000 chest X-ray images** annotated with 14 disease classes such as Atelectasis, Cardiomegaly, and Pneumothorax. This dataset ensures the model is trained on a diverse set of images to provide reliable predictions.

---

## Model Performance  
### Key Metrics  
- **Accuracy:** 89%  
- **Precision:** 88%  
- **Recall:** 92%  
- **F1-Score:** 91%
  
These metrics highlight the model's effectiveness in detecting and classifying chest diseases.

### Stability & Loss Curve  
![Loss Curve](./LossCurve.png)  
The loss curve demonstrates the model’s ability to converge during training, ensuring stable performance across different datasets.

### Example Predictions  
![Bounding Box Predictions](./multipleIpred.png)  
The model generates bounding boxes around areas of interest, with associated confidence scores, showcasing its capability to detect regions indicative of disease.

---

## Future Enhancements  
- **Real-Time Diagnostics:** Integrate APIs for real-time image analysis and diagnostics.  
- **Explainable AI:** Implement methods such as saliency maps to improve model interpretability.  
- **Fine-Grained Analysis:** Adopt Mask R-CNN for pixel-level segmentation of disease regions.  
- **Global Dataset Expansion:** Extend training to include more diverse datasets to cover rare diseases.  

---

## Getting Started  

### Installation  
1. **Clone the repository:**  
```bash  
git clone https://github.com/harshada-daund/Smart_Chest_X-Ray_Analysis.git  
cd Smart_Chest_X-Ray_Analysis  
```  
2. **Set up a virtual environment:**  
```bash  
python3 -m venv env  
source env/bin/activate  # On Windows, use `env\Scripts\activate`  
```  
3. **Install dependencies:**  
```bash  
pip install -r requirements.txt  
```  

### Usage  
1. **Run the application:**  
```bash  
python app.py  
```  
2. **Upload an X-ray image** via the provided interface.  
3. **View results** including detected diseases and confidence scores.  

---

## Contributing  
We welcome contributions from the community. To contribute:  
- Fork the repository.  
- Create a feature branch (`git checkout -b feature-branch`).  
- Commit your changes (`git commit -m 'Add new feature'`).  
- Push to the branch (`git push origin feature-branch`).  
- Open a pull request.  

---

By leveraging the power of deep learning, this project aims to assist healthcare professionals by automating chest disease detection, making diagnostic processes faster, more accurate, and more accessible.  

