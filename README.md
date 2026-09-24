# Explainable Breast Cancer Classification Using Mammography Images

An explainable deep learning framework for binary benign–malignant classification of mammography images using the CBIS-DDSM dataset.

The project compares a compact Convolutional Neural Network (CNN) baseline with an ImageNet-pretrained EfficientNetB0 model and uses Grad-CAM to inspect prediction-related image regions.

---

## Project Overview

Mammographic images can contain subtle abnormalities that are difficult to distinguish from normal breast tissue. Deep learning can learn image representations for classification, but the reasoning behind an individual prediction may not be directly visible.

This project investigates an explainable binary classification workflow consisting of:

1. CBIS-DDSM dataset acquisition
2. Dataset cleaning and validation
3. Image-path reconstruction
4. Duplicate handling
5. Patient-level train/validation/test splitting
6. Image preprocessing
7. Baseline CNN training
8. EfficientNetB0 transfer learning
9. Test-set evaluation
10. Model comparison
11. Grad-CAM-based visual explanation

The main research objectives are to develop a lightweight CNN baseline, compare it with EfficientNetB0, and examine model predictions using Grad-CAM.

---

## Dataset

The project uses the **Curated Breast Imaging Subset of the Digital Database for Screening Mammography (CBIS-DDSM)**.

After cleaning and reconstruction, the experimental dataset contains:

- **2,842 distinct images**
- **1,457 unique patients**
- **1,580 benign images**
- **1,262 malignant images**

Patient-level partitioning was used to prevent the same patient from appearing across training, validation, and test sets.

| Partition | Images | Patients |
|---|---:|---:|
| Training | 1,994 | 1,019 |
| Validation | 405 | 219 |
| Test | 443 | 219 |
| **Total** | **2,842** | **1,457** |

No patient overlap was reported between the three partitions.

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