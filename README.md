
![liver-tumor](https://www.aimspress.com/aimspress-data/era/2023/8/PIC/era-31-08-221-g001.jpg)


# **Liver Tumor Segmentation with U-Net**
This project was developed for the **DS304** course, focusing on **liver tumor segmentation** using the **U-Net** architecture.  
It applies deep learning to medical image analysis, with experiments comparing various hyperparameters.


---

## **Description**
The goal of this project is to accurately segment liver tumors from CT scans in the **LiTS dataset** using a **U-Net model**.  
Multiple experiments were conducted with different batch sizes, epochs, and learning rates to evaluate performance.

---

## **Technologies**
- **Data Processing**: NumPy, Pandas, SimpleITK, OpenCV
- **Data Visualization**: Matplotlib, Seaborn
- **Deep Learning**: PyTorch, Torchvision
- **Training Tools**: PyYAML, tqdm, TensorBoard
- **Metrics**: Dice Score, IoU

---

## **Dependencies**
All required packages are listed in `requirements.txt`.  
To install all dependencies, run:
```bash
pip install -r requirements.txt

---


## Dataset
The dataset consists of CT scan slices of the liver and corresponding tumor segmentation masks.

| **Folder**        | **Description**                          |
|-------------------|------------------------------------------|
| `data/raw/images` | CT scan images (.nii/.nii.gz)             |
| `data/raw/masks`  | Segmentation masks                        |

Dataset: [LiTS Challenge](https://competitions.codalab.org/competitions/17094)  
Reference Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## Performance
Below are the results for all experiments:

| **Batch Size** | **Epochs** | **Learning Rate** | **Dice Score** | **IoU Score** | **Training Time (min)** |
|----------------|------------|-------------------|----------------|---------------|-------------------------|
| 4              | 4          | 1e-5              | 0.5581         | 0.3871        | 1h24m30s                |
| 8              | 4          | 1e-5              | 0.8023         | 0.6699        | 1h24m37s                |
| 4              | 8          | 1e-5              | 0.8425         | 0.7279        | 2h43m14s                |
| 8              | 8          | 1e-5              | **0.8454**     | **0.7322**    | 2h38m17s                |
| 4              | 4          | 1e-4              | 0.8398         | 0.7239        | 1h22m52s                |
| 8              | 4          | 1e-4              | 0.8111         | 0.6822        | 1h21m27s                |
| 4              | 8          | 1e-4              | 0.8315         | 0.7116        | 2h44m16s                |
| 8              | 8          | 1e-4              | 0.7837         | 0.6444        | 2h39m26s                |

---

