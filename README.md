# Speech Emotion Detection

## Description

This project is part of my Master in Data Engieering (Semester 3) and more precisely, of the Speech Recognition module. It focuses on **Speech Emotion Detection**, leveraging **Machine Learning (ML) and Deep Learning (DL)** models to classify emotions from speech audio data. Various classifiers were tested to determine the most effective model for recognizing human emotions based on speech patterns.

A complete description of the project can be found in [this PDF](./ProjectASR.pdf) .

## Table of Contents

- [Dataset and Emotions](#dataset-and-emotions)
- [Preprocessing and Features Extraction](#preprocessing-and-features-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Model](#deep-learning-model)
- [Results and Best Model](#results-and-best-model)
- [Installation and Execution](#installation-and-execution)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

## Dataset and Emotions

The dataset used for this project is located in `EmotionalSpeechSet` and consists of labeled speech audio files. The dataset includes the following emotions:

- **Happy**
- **Sad**
- **Angry**
- **Neutral**
- **Disgust**
- **Fear**
- **Pleasant**

Each sample is linked to an audio file and its corresponding emotion label.

The dataset can be found on [my drive](https://drive.google.com/drive/folders/1MR3Bvx_OPnGNEtHqSmP_WtD0BiAvgcdp?usp=sharing).
It contains one folder and two files:
- `EmotionalSpeechSetBlind`: This folder contains all the audio files in wav format.
- `Train_audio_liste.csv`: This CSV file represents the training dataset and contains the data to be used for training and validation. It consists of 3 attributes:

  1. The index of the row
  2. `label`: The label of the data entry, which is the class or category of emotion
  3. `filepath`: The filepath, which represents the path of the audio file, related to the dataset folder.
- `BlindTest_audio_liste.csv`: This CSV file contains the data samples to be labeled to test our model.

## Preprocessing and Features Extraction

- **Checking for NULL values**:
  - The data was checked for NULL values, and there were none.
- **Resolving Inconsistency in Label Names**
  - All emotion labels were converted to lowercase to ensure consistency.
- **Data Balancing Analysis**
  - The dataset was analyzed for **class distribution** to check for imbalances in the emotions represented
  - The following table summarizes the number of occurrences for each label and their proportion in the dataset.

| Emotion  | Count | Percentage (%) |
|----------|-------|---------------|
| Neutral  | 292   | 14.90         |
| Angry    | 288   | 14.69         |
| Disgust  | 281   | 14.34         |
| Happy    | 281   | 14.34         |
| Sad      | 277   | 14.13         |
| Pleasant | 273   | 13.93         |
| Fear     | 268   | 13.67         |

The dataset appears to be relatively **balanced**, with no major discrepancies among the different emotion classes. The distribution ensures fair training across different categories, avoiding strong biases toward any particular emotion.

- **Features Extraction**:
  - **MFCCs (Mel-Frequency Cepstral Coefficients)** were extracted using **Librosa**.
- **Features Scaling**:
  - Features were standardized using **StandardScaler**.
- **Label Encoding**:
  - Emotion labels were converted into numerical categories using **LabelEncoder**.

## Machine Learning Models

The following ML models were trained and evaluated using **Stratified K-Fold cross-validation (5 splits):**

- **Naïve Bayes**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **MLP Classifier (Neural Network)**

## Deep Learning Model

A **Deep Learning approach (Feedforward Neural Network Classifier)** was also implemented using a **Keras Sequential Model**:

- **Architecture:**
  - Dense Layers with ReLU activation
  - Dropout for regularization
  - Output layer with Softmax for multi-class classification

## Results and Best Model

Each model was evaluated based on **Training Accuracy, Validation Accuracy, and Cross-Validation Scores**.


The feedforward neural network outperformed the other models, with a mean training accuracy of **0.9847** and validation accuracy around **0.9794**, showing strong generalization without overfitting. After evaluation on the test set, it achieved a final accuracy of **94%**, as confirmed by our instructor, further validating its effectiveness in real-world scenarios. It is the selected model.

 [Here](./Test_audio_labeled.csv) is the labeled test split.




## Installation and Execution

### Prerequisites

- **Python**
- **pip**
- **Virtual Environment (venv)**

### Clone the Repository

```sh
git clone https://github.com/Starias22/Speech-Emotion-Detection.git
cd Speech-Emotion-Detection
```

### Download the dataset

Go to [my drive](https://drive.google.com/drive/folders/1MR3Bvx_OPnGNEtHqSmP_WtD0BiAvgcdp?usp=sharing), dowload the dataset and unzip it

### Set up a Virtual Environment and Install Dependencies

```sh
python3 -m venv project_env
source project_env/bin/activate
pip install -r requirements.txt
```

### Execution

Run the notebook.

## Usage

- Modify **hyperparameters**, and **features extraction** methods for optimization.
- Experiment with **different ML/DL architectures**.

## Contributing

We welcome contributions! Follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## Contact Information

For questions or issues, please contact:

- **Name**: G. Ezéchiel ADEDE
- **Email**: [adedeezechiel@gmail.com](mailto:adedeezechiel@gmail.com)
- **GitHub**: [Starias22](https://github.com/Starias22)
- **LinkedIn**: [G. Ezéchiel ADEDE](https://www.linkedin.com/in/Starias22)

