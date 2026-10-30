# AI-Based Predictive Maintenance for Aircraft Engines
### Project Overview

This project uses Deep Learning to predict the Remaining Useful Life (RUL) of aircraft engines based on sensor readings. It uses the NASA C-MAPSS Turbofan Engine Degradation dataset to understand how engine conditions change over time and estimate how many operating cycles remain before the simulated end of life.

The project demonstrates the implementation of neural networks using PyTorch, time-series modeling, model evaluation, regularization techniques, and pretrained Transformer models.

## Objectives

- Understand PyTorch tensors, automatic differentiation, and neural network training.
- Predict aircraft engine Remaining Useful Life using sensor data.
- Implement and compare Artificial Neural Network (ANN), 1D Convolutional Neural Network (CNN), and Long Short-Term Memory (LSTM) models.
- Understand dropout, batch normalization, and early stopping.
- Evaluate models using MAE, RMSE, and R².
- Explore pretrained Transformer models through Hugging Face.

## Dataset

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation — FD001 subset

The dataset contains simulated engine operating cycles, operating conditions, and sensor measurements that reflect engine degradation.

**Dataset sources:**
- [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [C-MAPSS dataset files on GitHub](https://github.com/subhodeepmoitra/NASA---CMAPSSData)

### Dataset Features

- Engine unit identifier
- Operating cycle
- Three operating-condition settings
- Up to 21 sensor measurements
- Remaining Useful Life (RUL), calculated from the engine's degradation timeline

RUL represents the number of operating cycles remaining until the simulated end of an engine's useful life.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Cloud-based development and execution |
| PyTorch | Building and training deep learning models |
| NumPy | Numerical calculations |
| Pandas | Dataset loading and processing |
| Scikit-learn | Data scaling, splitting, and model evaluation |
| Matplotlib | Visualizing training and prediction results |
| Hugging Face Transformers | Demonstrating pretrained Transformer models |
| Joblib | Saving the data scaler |

## Models Implemented

### 1. Artificial Neural Network (ANN)

The ANN acts as a baseline model. It learns relationships between engine sensor readings and remaining useful life.

### 2. One-Dimensional Convolutional Neural Network (1D CNN)

The CNN identifies patterns across consecutive sensor readings and learns useful features from the time-series input.

### 3. Long Short-Term Memory (LSTM)

The LSTM is a recurrent neural network designed to learn sequential relationships. It uses sensor readings from previous operating cycles to estimate RUL.

## Project Workflow

1. Download the NASA C-MAPSS FD001 dataset.
2. Load and inspect engine sensor readings.
3. Calculate the RUL target for the training data.
4. Normalize sensor features using `StandardScaler`.
5. Split training and validation data by engine unit.
6. Convert sensor readings into sequences of 30 consecutive cycles.
7. Build and train ANN, CNN, and LSTM models using PyTorch.
8. Apply dropout, batch normalization, and early stopping where appropriate.
9. Evaluate predictions on the held-out test engines.
10. Compare models and visualize their results.
11. Save the best model, scaler, and model-comparison table.
12. Demonstrate pretrained Transformer-based text classification with Hugging Face.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual RUL.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more strongly.
- **R² Score:** Measures how well predictions explain the variation in actual RUL.

Lower MAE and RMSE generally indicate better predictions. An R² closer to 1 generally indicates a better fit.

Actual results are generated when the notebook runs.

## Visualizations

The notebook generates:

- Training and validation loss curves
- RMSE comparison between ANN, CNN, and LSTM
- MAE comparison between models
- Actual versus predicted RUL graph
- A table containing sample predictions and absolute errors

These visualizations help compare model performance and understand prediction accuracy.

## How to Run the Project

### Step 1: Open Google Colab

Visit [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Configure the runtime

Select **Runtime → Change runtime type** and choose a GPU if available.

### Step 3: Run the notebook

Paste the project code into a Colab cell and execute it.

The notebook downloads the dataset, preprocesses the sensor readings, trains the models, evaluates them, and generates visualizations.

### Step 4: Review the results

Compare the evaluation metrics and graphs to identify which model performs best on the held-out test engines.

### Step 5: Download generated files

The notebook saves the following files in the Colab runtime:

- `best_ann_model.pth`, `best_cnn_model.pth`, or `best_lstm_model.pth` — the saved weights of the best-performing model.
- `cmapss_scaler.pkl` — the fitted feature scaler.
- `model_comparison.csv` — the evaluation metrics for the three models.

The exact model filename depends on which model performs best. Download these files from the Colab Files panel if you want to keep them after the session ends.

## Project Structure

```text
aircraft-engine-predictive-maintenance/
│
├── README.md
├── aircraft_engine_prediction.ipynb
├── model_comparison.csv
├── cmapss_scaler.pkl
└── best_model.pth
```

*Note: The notebook and generated files must be added to the repository manually. The saved model filename may differ from the example above.*

## Expected Outcome

The project produces RUL predictions for previously unseen test engines and compares three deep learning approaches using numerical metrics and graphs.

It provides practical experience with neural network training, time-series data preparation, regularization, performance evaluation, and pretrained Transformer inference.

The best-performing model is selected based on the evaluation results generated by the notebook rather than assuming that a particular architecture will always perform best.

## Limitations

- The dataset is simulated and does not represent every real-world aircraft engine.
- Predictions depend on the selected sensor features, preprocessing, sequence length, and training configuration.
- The pretrained Transformer demonstration performs text classification; it is separate from the numerical RUL prediction task.
- The current implementation does not include a pretrained image CNN or a live OpenAI/Anthropic API integration.
- This is an educational research project and must not be used as a real aircraft safety or maintenance decision system.

## Future Scope

- Experiment with GRU and Transformer-based time-series models.
- Explore transfer learning using a suitable pretrained CNN on an image-based dataset.
- Integrate an LLM to explain model predictions in natural language.
- Develop a dashboard to visualize sensor trends and predicted RUL.
- Investigate uncertainty estimation and evaluation on additional C-MAPSS subsets.

## Conclusion

This project demonstrates how deep learning can be applied to predictive maintenance using engine sensor time-series data. By comparing ANN, CNN, and LSTM models, it illustrates the differences between basic neural networks, convolutional feature extraction, and sequential learning.

The project also provides a foundation for exploring pretrained models and more advanced AI applications.

**Disclaimer:** This project is intended for educational purposes only. Its predictions are not certified for operational aircraft maintenance.

