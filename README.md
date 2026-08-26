# Iris Species Classification with Logistic Regression

A beginner-friendly, end-to-end machine learning walkthrough that explores the classic **Iris flower dataset** and builds a **Logistic Regression** model to classify iris flowers into their correct species based on sepal and petal measurements.

## 📌 Project Overview

This project demonstrates a complete (small-scale) machine learning workflow:

1. Load and inspect a dataset
2. Clean/prepare the data for modeling
3. Explore the data visually (EDA)
4. Split data into training and testing sets
5. Train a classification model
6. Evaluate the model's accuracy
7. Use the trained model to make a prediction on new, unseen data

The goal is educational — each code cell is paired with a plain-English explanation of *what* it does and *why*, making it a good reference for anyone learning the basics of data science and machine learning in Python.

## 📊 Dataset

**[Iris Dataset](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset)** — a well-known dataset in machine learning containing 150 samples of iris flowers, with 4 measurements per flower:

| Column | Description |
|---|---|
| `sepal_length` | Length of the sepal (cm) |
| `sepal_width` | Width of the sepal (cm) |
| `petal_length` | Length of the petal (cm) |
| `petal_width` | Width of the petal (cm) |
| `species` | Flower species: *setosa*, *versicolor*, or *virginica* |

> **Note:** To run this notebook yourself, download `iris.csv` from the Kaggle link above and place it in the same folder as the notebook.

## 🛠️ Tools & Libraries

- **pandas** – loading and organizing tabular data
- **NumPy** – numerical operations
- **Matplotlib** & **Seaborn** – data visualization
- **scikit-learn** – train/test splitting and Logistic Regression modeling

## 🔍 Workflow Breakdown

### 1. Data Loading & Inspection
The dataset is loaded into a pandas DataFrame using `pd.read_csv()`. `iris.head()` and `iris.info()` are used to preview the data and confirm there are no missing values and that column types are correct (150 rows, 5 columns).

### 2. Data Preparation
The categorical `species` column (text labels: *setosa*, *versicolor*, *virginica*) is converted into numeric labels (`0`, `1`, `2` respectively), since machine learning models require numeric input.

### 3. Exploratory Data Analysis (EDA)
Several visualizations are used to understand relationships in the data:
- **Pair plot** (`sns.pairplot`) — shows relationships between every pair of measurements, split by species.
- **Bar chart** — compares average sepal length across species (virginica has the largest average, setosa the smallest).
- **Box plots** — show the spread/quartiles of sepal width and petal length per species, revealing that setosa has noticeably smaller petals than the other two species.

### 4. Train/Test Split
The features (`X`: the four measurements) are separated from the target (`y`: species label). The data is then split into **70% training** and **30% testing** sets using `train_test_split` (with a fixed `random_state` for reproducibility).

### 5. Model Training
A **Logistic Regression** classifier (`max_iter=500`) is trained on the training data to learn the relationship between flower measurements and species.

### 6. Evaluation
The trained model predicts species on the held-out test set. Model accuracy is calculated with `.score()`.

**Result: ~97.8% accuracy** on the test set.

### 7. Prediction on New Data
A brand-new, hand-crafted flower measurement is fed into the trained model to demonstrate real-world usage — the model successfully predicts which species the new sample most likely belongs to.

## 📈 Results

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| Train/Test Split | 70% / 30% |
| Test Accuracy | ~97.8% |

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Download `iris.csv` from [Kaggle](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset) and place it in the project folder
4. Open and run the notebook:
   ```bash
   jupyter notebook iris_classification.ipynb
   ```

## 🧠 Key Takeaways

- How to load, clean, and encode a dataset for machine learning
- How to use visualizations to spot patterns before modeling
- How to properly split data into training/testing sets
- How to train and evaluate a simple classification model
- How to use a trained model to make predictions on new data

## 📄 License

This project is for educational purposes and uses a publicly available dataset from Kaggle.
