# Skin Cancer Detection
![image](https://github.com/SRDdev/Skin_Cancer_Detection/assets/84516626/ebee7572-8940-403b-9608-9aa65a6f94c8)

## Overview

This repository focuses on utilizing Convolutional Neural Networks (CNNs) implemented in TensorFlow for the crucial task of skin cancer detection. Skin cancer is a prevalent form of cancer, and early detection plays a pivotal role in effective treatment. The repository provides two distinct types of CNN models: TensorFlow Functional and TensorFlow Sequential.

```
Note: This repostory was created in early 2022 and is now open-sourced.
```


## Models

### 1. TensorFlow Functional Model

The TensorFlow Functional Model leverages the functional API of TensorFlow, offering a flexible and customizable approach to building CNN architectures for skin cancer detection. This model allows for the creation of complex network structures with shared layers and multiple inputs/outputs.

### 2. TensorFlow Sequential Model

The TensorFlow Sequential Model follows a linear stack of layers, simplifying the process of building CNNs for skin cancer detection. Sequential models are suitable for straightforward architectures, making them user-friendly and efficient for quick experimentation.

## Key Features

- **CNN Architectures:** Both models are designed specifically for skin cancer detection, incorporating convolutional layers, pooling, and fully connected layers to capture intricate patterns in dermatoscopic images.

- **Transfer Learning:** Take advantage of pre-trained models like VGG16, ResNet, or MobileNet as a starting point for training, enhancing the detection capabilities even with limited labeled data.

- **Data Augmentation:** Mitigate overfitting and improve model generalization by applying data augmentation techniques to artificially increase the diversity of the training dataset.

- **Evaluation Metrics:** Evaluate the model performance using standard metrics such as accuracy, precision, recall, and F1 score to ensure robust and reliable skin cancer detection.

## Getting Started

1. **Clone the Repository:** Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/SRDdev/Skin-Cancer-Detection.git
   ```

2. **Install Dependencies:** Check the `requirements.txt` file for necessary dependencies and install them using:
   ```
   pip install -r requirements.txt
   ```

3. **Dataset Preparation:** Organize your skin cancer dataset following the specified directory structure in the documentation.

4. **Model Training:** Use the provided notebooks or scripts to train the TensorFlow Functional or TensorFlow Sequential model on your dataset.

## Usage

Explore the notebooks and code examples to understand how to load the models, make predictions, and fine-tune them according to your specific requirements.

## Contributions

Contributions to the Skin Cancer Detection repository are welcome. Whether it's optimizing existing models, adding new features, or improving documentation, your contributions can make a significant impact in the fight against skin cancer.

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We extend our appreciation to the open-source community, healthcare professionals, and researchers whose collective efforts contribute to advancements in skin cancer detection.

Let's work together to make a positive impact on skin cancer diagnosis using the power of CNNs in TensorFlow!
