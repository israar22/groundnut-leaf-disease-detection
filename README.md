# Groundnut Leaf Disease Detection and Classification

A deep-learning project that classifies groundnut (peanut) leaf images into five categories using **EfficientNetV2B0 transfer learning**.

> This repository is a reproducible implementation of the methodology described in the project report. The report states validation metrics of 96.51% accuracy, 96.56% precision, 96.51% recall and 96.48% F1-score. These are reported results, not results independently reproduced by this repository. Re-training may produce different metrics depending on the dataset, split, random seed, software versions and hardware.

## Classes

1. `ALTERNARIA LEAF SPOT`
2. `HEALTHY`
3. `LEAF SPOT (EARLY AND LATE)`
4. `ROSETTE`
5. `RUST`

The class names are intentionally kept consistent with the project report and its alphabetical directory order.

## Methodology

- Dataset: 1,720 field-captured JPG images (as described in the report).
- Source location described in report: Ramchandrapur village, Purba Medinipur, West Bengal, India.
- Input size: 224 × 224 RGB.
- Split: 80% training / 20% validation.
- Augmentation: random horizontal/vertical flips, rotation, zoom and contrast.
- Backbone: ImageNet-pretrained EfficientNetV2B0.
- Classification head: Global Average Pooling, Batch Normalization, Dropout (0.3), Dense Softmax.
- Training: 15 frozen-backbone epochs, followed by 10 fine-tuning epochs (top layers unfrozen from layer 100).

## Repository layout

```text
groundnut-leaf-disease-detection/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   └── model_utils.py
├── data/
│   └── groundnut_leaf_dataset/
│       ├── ALTERNARIA LEAF SPOT/
│       ├── HEALTHY/
│       ├── LEAF SPOT (EARLY AND LATE)/
│       ├── ROSETTE/
│       └── RUST/
├── models/       # trained model and class_names.json are generated here
└── outputs/      # plots and evaluation reports are generated here
```

## 1. Set up the environment

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/<your-username>/groundnut-leaf-disease-detection.git
cd groundnut-leaf-disease-detection

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

TensorFlow installation can vary by operating system and GPU. If installation fails, follow the official TensorFlow installation guide for your platform and then install the remaining packages.

## 2. Add the dataset

The original image dataset is not included in this repository. Add it locally under:

```text
data/groundnut_leaf_dataset/
```

Each class must have its own folder, and each folder must contain images:

```text
data/groundnut_leaf_dataset/
├── ALTERNARIA LEAF SPOT/
│   ├── image001.jpg
│   └── ...
├── HEALTHY/
├── LEAF SPOT (EARLY AND LATE)/
├── ROSETTE/
└── RUST/
```

Folder names must match exactly. The training script checks for the five expected folders.

**Dataset rights and privacy:** Before publishing any images, confirm that you have permission to redistribute the dataset. If you do not, keep the image files out of GitHub and provide dataset acquisition instructions or a source link that you are authorized to share. Do not upload private or restricted data.

## 3. Train the model

From the repository root:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --data_dir data/groundnut_leaf_dataset --epochs 15 --fine_tune_epochs 10 --batch_size 32
```

The script creates:

- `models/groundnut_efficientnetv2b0.keras`
- `models/class_names.json`
- `outputs/training_history.png`
- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`

Training uses an 80/20 directory split. The script prints class-wise precision, recall and F1-score plus weighted averages.

## 4. Predict one image from the command line

After training:

```bash
python predict.py --image path/to/groundnut_leaf.jpg
```

Specify a different model if needed:

```bash
python predict.py --image path/to/leaf.jpg --model models/groundnut_efficientnetv2b0.keras
```

The prediction includes the predicted class and model confidence. Confidence is the model's softmax score; it should not be interpreted as a calibrated probability or a confirmed agricultural diagnosis.

## 5. Run the web application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a JPG/PNG leaf image, and view the predicted class and class scores.

The app requires a trained model at `models/groundnut_efficientnetv2b0.keras`. Train the model first and keep `models/class_names.json` beside it.

## Evaluation and interpretation

The project report records the following validation metrics:

| Metric | Reported value |
|---|---:|
| Accuracy | 96.51% |
| Weighted precision | 96.56% |
| Weighted recall | 96.51% |
| Weighted F1-score | 96.48% |
| Specificity | 98.77% |
| AUC-ROC | 99.35% |

The report also notes that the Rosette class had fewer validation examples (17) and a lower class recall (82.35%). A small or imbalanced class can be harder to learn. Evaluate the model on a separate field dataset before using it outside a demonstration or research setting.

## Important limitations

- This is an image classification prototype, not a replacement for diagnosis by an agricultural expert.
- Predictions may be unreliable for unfamiliar cultivars, lighting, backgrounds, disease stages or camera conditions.
- The dataset is relatively small and collected from a stated field location; external validation is needed before claiming broad generalization.
- The model predicts one of the five labels. It does not estimate disease severity, prescribe pesticides or provide treatment advice.
- Do not make crop-management decisions from a prediction alone.

## Reproducibility notes

- The class order is explicitly fixed to avoid label-order mismatches between training and inference.
- The training script sets a random seed, but exact reproducibility is not guaranteed across TensorFlow versions and hardware.
- The training/validation split is generated by TensorFlow's directory loader. For publication-grade evaluation, consider a separate held-out test set and grouping images by plant/plot/session to reduce near-duplicate leakage.

## Acknowledgement

Project: **Groundnut Leaf Disease Detection and Classification**  
Original project report: Team 11, Technical Answers for Real World Problems (TARP), Integrated M.Tech, Computer Science and Engineering, VIT Vellore, November 2025.

## References

The project report cites, among others:

1. B. Sasmal et al., “A novel groundnut leaf dataset for detection and classification of groundnut leaf diseases,” *Data in Brief*, vol. 55, 110763, 2024. DOI: [10.1016/j.dib.2024.110763](https://doi.org/10.1016/j.dib.2024.110763).
2. H. Kukadiya et al., “Effective groundnut crop management by early prediction of leaf diseases through convolutional neural networks,” *International Research Journal of Multidisciplinary Technovation*, 2024. DOI: [10.54392/irjmt2412](https://doi.org/10.54392/irjmt2412).
3. Suresh and K. Seetharaman, “Real-time automatic detection and classification of groundnut leaf disease using hybrid machine learning techniques,” *Multimedia Tools and Applications*, 2023. DOI: [10.1007/s11042-022-12893-1](https://doi.org/10.1007/s11042-022-12893-1).
4. H. Qi et al., “Automatic identification of peanut-leaf diseases based on stack ensemble,” *Applied Sciences*, 2021. DOI: [10.3390/app11041950](https://doi.org/10.3390/app11041950).
5. N. Paramanandham et al., “Enhancing disease detection with weight initialization and residual connections using LeafNet for groundnut leaf diseases,” *IEEE Access*, 2024. DOI: [10.1109/ACCESS.2024.3422311](https://doi.org/10.1109/ACCESS.2024.3422311).

See the original report for the complete bibliography.

## License

Choose a license only after confirming that you have the right to publish the code and any included assets. The dataset is not bundled with this repository.
