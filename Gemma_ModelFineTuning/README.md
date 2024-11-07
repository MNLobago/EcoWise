# Project Title: Model Training and Fine-Tuning Repository

## Overview
This repository contains multiple versions of a model focused on reducing the carbon footprint. Each version features organized directories and relevant files to assist in understanding and utilizing the model effectively.

## Directory Structure

```
├── MODELV1/
│   ├── fine-tuning-gemma-2-model-using-lora-and-kerasMNL.ipynb
│   ├── gemma_llm_model.json
│   ├── model_architecture.png
│  
│
├── MODELV2/
│   ├── fine-tuning-gemma-2-model-using-lora-and-kerasMNL2.ipynb
│   ├── gemma_llm_model.json
│   ├── model_architecture.png
│  
│
├── MODELV3/
│   ├── fine-tuning-gemma-2-model-using-lora-and-kerasMNL3.ipynb
│   ├── gemma_llm_model.json
│   ├── model_architecture.png
│ 
│
└── high_volume_carbon_footprint_qa.json
```

## Model Versions
- **MODELV1**: Initial version of the model with basic features and fine-tuning applied.
- **MODELV2**: Initial experimentation with hyperparameter tuning and fine-tuning of the Gemma2 2B model.
- **MODELV3**: Second experimentation with hyperparameter tuning and fine-tuning of the Gemma2 2B model.

## Key Files
- **fine-tuning-gemma-2-model-using-lora-and-kerasMNL*.ipynb**: Jupyter Notebook detailing the fine-tuning process of the Gemma-2 model using LoRA and Keras, including training steps and evaluation procedures.
- **gemma_llm_model.json**: JSON file containing the model architecture and parameters, facilitating loading and evaluation of the trained model.
- **model_architecture.png**: Diagram illustrating the architecture of the Gemma LLM model for simplified visualization and understanding.

## Training Data
- **high_volume_carbon_footprint_qa.json**: The training data used for modeling, which contains high-volume carbon footprint data points essential for training and testing the models.

## Usage Instructions
To run the fine-tuning notebooks:
1. Clone the repository to your local machine.
2. Navigate to the desired model version folder (MODELV1, MODELV2, or MODELV3).
3. Open the Jupyter Notebook (`fine-tuning-gemma-2-model-using-lora-and-kerasMNL*.ipynb`) and follow the instructions within to execute the cells and fine-tune the model.
