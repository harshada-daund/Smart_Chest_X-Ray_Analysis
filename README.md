<h1 align="center">🩺 Chest Disease Detection using Deep Learning (R-CNN)</h1>  
<p align="center"><i>Accelerating Diagnosis with Artificial Intelligence</i></p>  

---

<h2>✨ Description</h2>  
<p align="justify">  
The <b>Chest Disease Detection using Deep Learning (R-CNN)</b> project automates the detection and classification of chest diseases from X-ray images using Faster R-CNN architecture. It assists radiologists in diagnosing conditions like pneumonia, tuberculosis, and lung cancer. The project leverages the NIH ChestX-ray14 dataset, providing annotated data for robust training and evaluation.  
</p>  

---

<h2>🎯 Objectives</h2>  
<p align="justify">  
- Design a Faster R-CNN-based model for thoracic pathology detection.<br>  
- Evaluate model performance using precision, recall, F1-score, and accuracy.<br>  
- Apply data augmentation and transfer learning to improve accuracy.<br>  
- Provide interpretable outputs through bounding box predictions.<br>  
</p>  

---

<h2>🛠️ Technologies Used</h2>  
<p align="justify">  
- <b>Frameworks:</b> PyTorch, Detectron2<br>  
- <b>Visualization Tools:</b> Matplotlib, Seaborn<br>  
- <b>Model:</b> Faster R-CNN with ResNet-50 backbone<br>  
- <b>Dataset:</b> NIH ChestX-ray14<br>  
- <b>Utilities:</b> OpenCV, Google Colab<br>  
</p>  

---

<h2>📂 Dataset</h2>  
<p align="justify">  
The <b>NIH ChestX-ray14</b> dataset contains over 112,000 annotated chest X-ray images across 14 disease classes, such as Atelectasis, Cardiomegaly, and Pneumothorax.  
</p>  

---

<h2>📈 Results</h2>  

<h3>1️⃣ Performance Metrics</h3>  
<p align="justify">  
The model achieved strong performance metrics across multiple disease classes:  
</p>  
<ul>
  <li><b>F1-Score:</b> <mark>91%</mark></li>
  <li><b>Accuracy:</b> <mark>89%</mark></li>
  <li><b>Precision:</b> <mark>88%</mark></li>
  <li><b>Recall:</b> <mark>92%</mark></li>
</ul>  
<p align="justify">  
These metrics demonstrate the model's ability to accurately detect chest diseases while balancing false positives and false negatives.
</p>  

<h3>2️⃣ Loss Curve</h3>  
<p align="center">
<img src="./LossCurve.png" alt="Loss Curve" width="700">
</p>  
<p align="justify">  
The loss curve tracks the model's convergence over epochs, showcasing its stability during training and validation.  
</p>

<h3>3️⃣ Example Predictions with Confidence Scores</h3>  
<p align="center">
<img src="./multipleIpred.png" alt="Bounding Box Predictions" width="700">
</p>  
<p align="justify">  
The bounding boxes represent detected regions of interest with associated confidence scores, demonstrating the model's localization capabilities.  
</p>

<h3>4️⃣ Prediction Table</h3>  
<p align="center">
<img src="./ImagePredictionTable.png" alt="Prediction Table" width="700">
</p>  
<p align="justify">  
This table provides quantitative insights into the model's predictions for sample X-rays, including bounding box coordinates and confidence scores.  
</p>

---

<h2>🌟 Future Scope</h2>  
<p align="justify">  
- <b>Real-Time Data:</b> Integrate APIs for live diagnostics.<br>  
- <b>Explainable AI:</b> Enhance model interpretability using saliency maps.<br>  
- <b>Pixel-Level Segmentation:</b> Adopt Mask R-CNN for fine-grained segmentation.<br>  
- <b>Global Dataset Expansion:</b> Include rare diseases and multi-institutional data.<br>  
</p>  

---
