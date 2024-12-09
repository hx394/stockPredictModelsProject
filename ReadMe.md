# Project: Comparison Among GRU, LSTM, and Arima/Garch Hybrid Models to Predict Stock Price

Contributors: Hongzhen Xu, Cheyi Wu, Jingming Cheng

Project Overview: 

Using GRU, LSTM, GRU-Garch Hybrid, LSTM-Garch Hybrid, LSTM-Arima Hybrid models to predict stock price.

Applying plots and statistics among compare groups to evaluate the various approaches and models.

-------------------------------------------------------------------------------------------------------

## Space for Hongzhen Xu

### who implemented the GRU rate stock predict model and LSTM rate stock predict model

There are 3 folders in my space, "data preprocessing code", "input data", "models and results".

Instructions for GRU and LSTM models(accomplished by Hongzhen Xu):

When run the code, make sure the paths to the files are correct.

The GRU rate model and LSTM rate model were trained on Google Colab, 

but the models were not saved because of the capacity.

If you would like to run the deep learning code in an environment other than Google Colab, 

you need to change the code a little bit to read input files and write output files. 

Make sure you have Python 3.8+, the following dependencies are required:

-torch

-torchvision

-torch.nn

-torch.optim 

-numpy

-pandas

-sklearn.preprocessing

-sklearn.model_selection

-torch.utils.data

-matplotlib.pyplot

-tqdm

-----------------------------------------------------------------------------------------------------------

## Space for Cheyi Wu

### who implemented the GRU-GARCH model and LSTM-GARCH model

This part implements and compares different hybrid approaches combining deep learning models (GRU/LSTM) with GARCH (Generalized Autoregressive Conditional Heteroskedasticity) for stock price prediction. The models were tested on seven major stocks: AAPL, IBM, META, MSFT, NVDA, TSLA, and VZ.

### Models Implemented
- Base Models:
  - GRU (Gated Recurrent Unit) (Based on Hongzhen's work)
  - LSTM (Long Short-Term Memory) (Based on Hongzhen's work)
-GARCH(1,1)
- Hybrid Models:
  - GRU-GARCH
  - LSTM-GARCH

#### Model Architectures
GRU-GARCH Hybrid
```
class GRU_GARCH(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.omega = nn.Parameter(torch.tensor([0.1]))
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.beta = nn.Parameter(torch.tensor([0.8]))
```
LSTM-GARCH Hybrid
```
class LSTM_GARCH(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.omega = nn.Parameter(torch.tensor([0.1]))
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.beta = nn.Parameter(torch.tensor([0.8]))
```
#### Implementation Details
- Framework: PyTorch
- Training Device: GPU (CUDA) when available
- Optimization: Adam optimizer
- Loss Function: Mean Squared Error (MSE)
- Evaluation Metrics: MSE, MAE, RMSE

#### Special Cases (Run and only Garch-NVDA.ipynb for NVDA's stock for GRU-Garch model)
NVDA: Required modified architecture due to high price volatility
Increased hidden dimensions
Additional layers
Modified learning rate
Enhanced error handling

#### Output Files
- model_comparison.csv: Comparative metrics for base and hybrid models
- predictions.csv: Actual vs predicted prices
- training_loss.png: Training loss curves
- predictions.png: Visualization of predictions

#### Required Libraries
```
# Core libraries
pip install numpy
pip install pandas
pip install torch  # PyTorch
pip install scikit-learn
pip install matplotlib

# GARCH modeling
pip install arch  # For ARCH/GARCH models

# Data processing and visualization
pip install seaborn
pip install plotly  # Optional, for interactive plots
```
#### Version Requirements
-numpy>=1.19.2
- pandas>=1.2.0
- torch>=1.9.0
- scikit-learn>=0.24.2
- matplotlib>=3.3.4
- arch>=5.0.0
- seaborn>=0.11.2
- plotly>=5.3.1  # Optional
#### Environment Setup
```
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate  # For Windows

# Install all requirements
pip install -r requirements.txt
```
#### GPU Support (Optional but Recommended)
```
# For CUDA support (check PyTorch website for correct version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```
#### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)
- Minimum 8GB RAM
- 50GB free disk space (for data and model storage)

---------------------------------------------------------------------------------------------------------

## Space for Jingming Cheng

### who implemented the Arima-LSTM model and evaluation metrics codes

This part is focused on time series analysis and forecasting using a combination of classical statistical methods (e.g., ARIMA) and deep learning models (e.g., LSTM). The objective is to explore, analyze, and predict time-dependent data, leveraging tools for preprocessing, modeling, and performance evaluation.

#### Note
-   This section uses Python 3.10.11, TensorFlow 2.18.0, Keras 3.7.0, some functions may not be supported on Windows 7 or earlier systems.
-   For data synthesize part please visit my own GitHub repository https://github.com/jerry0012000/CS6140ProjectTask1_Data_Synthesize
-   In Analyze GRU&LSTM folder, original data is from Hongzhen Xu folder, please check BASE_DIR variable.
-   In Jingming Cheng/result/[stockname] folder, only arima_forecast_results_[stockname].xlsx and arima_predict_result.png are files generated by arima_stock_analysis.ipynb. Other files in the same folder are generated by arima-lstm.ipynb, diff1.png and diff2.png are result after first and second order difference.
-   To run program on different stock, please change stock parameter in Jupyter Notebooks.
-   Only the arima_old.ipynb file uses the auto_arima function. Since it is outdated and lacks certain details, such as the 95% confidence interval and visualizations of some calculation processes, the output directory for this file was not updated and is currently disabled. Furthermore, this file does not align with the current project standards and will likely be deprecated in future updates.


#### Environment and Libraries

The part uses following key libraries:

1. Data Manipulation and Visualization


- **pandas**: For data manipulation and analysis.

- **numpy**: For numerical computations.

- **matplotlib.pyplot** and **seaborn**: For data visualization.

2. Statistical Analysis

- **scipy.stats**: For statistical testing and data transformations.

- **statsmodels**:
  - `ARIMA`: Used for modeling time series data with the ARIMA method.
  - `adfuller`: For stationarity testing (Augmented Dickey-Fuller test).
  - `plot_acf` and `plot_pacf`: For plotting autocorrelation and partial autocorrelation functions.
  - `acorr_ljungbox`: For residual diagnostics and autocorrelation testing.

3. Machine Learning

- **scikit-learn**:
  - `MinMaxScaler`: For scaling data into a range suitable for machine learning models.
  - `mean_squared_error`, `mean_absolute_error`, and `r2_score`: For model evaluation metrics.

4. Deep Learning

- **TensorFlow (v2.18.0)** and **Keras (v3.7.0)**:
  - `Sequential`: For building neural networks.
  - `Dense`, `Dropout`, and `LSTM`: Core layers used in constructing and training the LSTM model for time series forecasting.



#### Features of the Project

1. **Time Series Preprocessing**:
   - Handles stationarity testing and transformations (e.g., differencing).
   - Scales data for deep learning models using Min-Max normalization.
2. **Statistical Modeling**:
   - Implements ARIMA for baseline time series forecasting.
   - Provides diagnostic tools like autocorrelation checks and residual analysis.
3. **Deep Learning Forecasting**:
   - Utilizes LSTM networks for sequence-to-sequence prediction.
   - Incorporates dropout layers to prevent overfitting.
4. **Model Evaluation**:
   - Uses metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score to assess model performance.
5. **Visualization**:
   - Visualizes trends, seasonality, autocorrelations, and prediction results.



#### Requirements

To replicate this project, ensure the following libraries are installed:

    pip install pandas numpy matplotlib seaborn scipy statsmodels sklearn tensorflow

You also need Python 3.10.11 or later to ensure compatibility.



#### How to Use

1.Data Preparation:

- Load your time series data using pandas.

- Preprocess the data to ensure it is stationary (if needed) and properly scaled.

2.Modeling:

- Experiment with ARIMA to establish a baseline forecast.

- Train an LSTM model using TensorFlow for deep learning-based predictions.

3.Evaluation:

- Assess model performance using metrics like MSE, MAE, and R².

4.Visualization:

- Plot the predictions against the actual data to evaluate the model visually.



#### Acknowledgment

This part combines statistical and machine learning techniques to handle time series forecasting challenges efficiently, offering both simplicity and power in its modeling approaches.
