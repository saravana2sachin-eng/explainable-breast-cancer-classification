# Explainable Breast Cancer Classification Using Mammography Images

This project investigates deep learning and explainable AI (XAI) for breast cancer classification using mammography images from the CBIS-DDSM dataset.

The project compares a custom baseline Convolutional Neural Network (CNN) with an ImageNet-pretrained EfficientNetB0 model and uses Grad-CAM to provide visual explanations of model predictions.

---

## Dataset

The study uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset.

After data cleaning and preprocessing:

- **2,842** distinct mammography images
- **1,457** patients
- **1,580** benign images
- **1,262** malignant images

### Patient-Level Data Split

| Split | Images | Patients |
|---|---:|---:|
| Training | 1,994 | 1,019 |
| Validation | 405 | 219 |
| Testing | 443 | 219 |

Patient-level splitting was used to prevent images from the same patient appearing across different splits.

---

## Preprocessing

The mammography images are processed using the following pipeline:

- Decode images as grayscale
- Resize images to **224 × 224**
- Convert pixel values to floating-point representation
- Replicate grayscale images into **3 channels**
- Use a batch size of **32**
- Shuffle the training data
- Use TensorFlow prefetching

The resulting image tensor has the shape:

```text
224 × 224 × 3
```

---

## Models

### 1. Baseline CNN

A custom convolutional neural network was developed as the baseline model.

Architecture:

- Conv2D: 32 filters
- Max Pooling
- Conv2D: 64 filters
- Max Pooling
- Conv2D: 128 filters
- Max Pooling
- Conv2D: 256 filters
- Global Average Pooling
- Dropout
- Dense layer with 128 units
- Dropout
- Sigmoid output

Total trainable parameters:

**421,441**

### 2. EfficientNetB0

EfficientNetB0 with ImageNet pretrained weights was used as a transfer-learning model.

Configuration:

- Input size: **224 × 224 × 3**
- ImageNet pretrained backbone
- Frozen backbone
- Classification head with sigmoid output

Total parameters:

**4,213,668**

Trainable parameters:

**164,097**

---

## Training Configuration

Both models were trained using:

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1 × 10⁻⁴ |
| Loss | Binary Cross Entropy |
| Epochs | 10 |
| Batch Size | 32 |
| Model Selection | Validation AUC |

---

## Results

### Baseline CNN

| Metric | Result |
|---|---:|
| Accuracy | 0.6185 |
| AUC | 0.6261 |
| Precision | 0.6552 |
| Recall | 0.2908 |
| F1 Score | 0.4028 |

### EfficientNetB0 vs Baseline CNN

| Metric | Baseline CNN | EfficientNetB0 |
|---|---:|---:|
| Accuracy | 0.6185 | 0.6208 |
| AUC | 0.6261 | 0.6670 |
| Precision | 0.6552 | 0.6228 |
| Recall | 0.2908 | 0.3622 |
| F1 Score | 0.4028 | 0.4581 |

---

## Result Visualizations

### Baseline CNN

The baseline model training and evaluation plots are available in:

[`results/baseline_cnn/`](results/baseline_cnn/)

Available visualizations include:

- Training accuracy
- Training AUC
- Binary cross-entropy loss
- Confusion matrix

### Model Comparison

Comparison results are available in:

[`results/model_comparison/`](results/model_comparison/)

### Explainable AI / Grad-CAM

Grad-CAM visualizations are available in:

[`results/xai/`](results/xai/)

The examples include:

- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive (TP)

---

## Confusion Matrix

For the baseline CNN test set:

| | Predicted Benign | Predicted Malignant |
|---|---:|---:|
| Actual Benign | 217 | 30 |
| Actual Malignant | 139 | 57 |

---

## Explainability

Grad-CAM was applied to visualize the regions of mammography images that contributed to model predictions.

The XAI analysis includes examples corresponding to:

- True Negative predictions
- False Positive predictions
- False Negative predictions
- True Positive predictions

The Grad-CAM analysis in this project is **qualitative** and does not constitute quantitative lesion-localization validation.

---

## Repository Structure

```text
explainable-breast-cancer-classification/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── breast_cancer_xai.ipynb
│
├── docs/
│   └── research_paper.pdf
│
├── results/
│   ├── baseline_cnn/
│   ├── model_comparison/
│   └── xai/
│
└── src/
```

---

## Research Paper

The research paper associated with this project is available here:

[Research Paper](docs/research_paper.pdf)

**Title:**  
*Explainable Deep Learning for Breast Cancer Classification Using Mammography Images*

---

## Reproducibility

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

Open:

```text
notebooks/breast_cancer_xai.ipynb
```

The dataset itself is not included in this repository.

---

## Limitations

The current study has several limitations:

- Evaluation is based on a single dataset.
- Malignant-class recall remains relatively modest.
- Grad-CAM evaluation is qualitative rather than quantitative.
- The EfficientNetB0 backbone was frozen during training.
- Additional data augmentation and class-imbalance strategies were not extensively evaluated.

---

## Future Work

Potential future improvements include:

- Data augmentation
- Class-imbalance handling
- EfficientNet fine-tuning
- Evaluation on an independent dataset
- Quantitative Grad-CAM / lesion localization evaluation
- Further optimization of malignant-class recall

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Disclaimer

This project is intended for academic and research purposes only. The models presented here are not medical diagnostic tools and should not be used for clinical decision-making.