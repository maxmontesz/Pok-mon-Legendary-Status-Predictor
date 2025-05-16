# Pok-mon-Legendary-Status-Predictor
# Pokémon Legendary Status Predictor

## Description
This project uses machine learning to predict whether a Pokémon is legendary based on its characteristics. It employs a Decision Tree Classifier to analyze various Pokémon features and determine their legendary status.

## Dataset
The project uses the "1025 Pokemon" dataset from Kaggle, which contains comprehensive information about various Pokémon.

Dataset source: [1025 Pokemon](https://www.kaggle.com/datasets/sarahtaha/1025-pokemon)

## Features
The model considers the following features to make predictions:
- Name
- Primary Typing
- Generation
- Form
- Evolution Stage
- Weight (hg)
- Height (in)
- Weight (lbs)
- Base Stat Total
- Health
- Attack
- Defense
- Special Attack
- Special Defense
- Speed

## Model
The project uses a Decision Tree Classifier from scikit-learn to predict the legendary status of Pokémon.

## Results
The model's performance is evaluated using accuracy, confusion matrix, and a classification report.

## Usage
1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook or Python script

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- kagglehub
- google-cloud-aiplatform

## Future Work
- Experiment with other machine learning algorithms
- Fine-tune hyperparameters for better performance
- Deploy the model using Google Cloud AI Platform

Feel free to contribute to this project by submitting pull requests or opening issues for any bugs or feature requests.
