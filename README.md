
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
