Project: Comparison Among GRU, LSTM, and Arima/Garch Hybrid Models to Predict Stock Price

Contributors: Hongzhen Xu, Cheyi Wu, Jingming Cheng

Project Overview: 

Using GRU, LSTM, GRU-Garch Hybrid, LSTM-Garch Hybrid, LSTM-Arima Hybrid models to predict stock price.

Applying plots and statistics among compare groups to evaluate the various approaches and models.

-------------------------------------------------------------------------------------------------------

## Space for Hongzhen Xu

### who Implemented the GRU rate stock predict model and LSTM rate stock predict model

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

(待则谊补充)

---------------------------------------------------------------------------------------------------------

## Space for Jingming Cheng

### who Implemented the Arima-LSTM model and evaluation metrics codes

This part is focused on time series analysis and forecasting using a combination of classical statistical methods (e.g., ARIMA) and deep learning models (e.g., LSTM). The objective is to explore, analyze, and predict time-dependent data, leveraging tools for preprocessing, modeling, and performance evaluation.

#### Note
-   This section uses Python 3.10.11, TensorFlow 2.18.0, Keras 3.7.0, some functions may not be supported on Windows 7 or earlier systems.
-   For data synthesize part please visit my own GitHub repository https://github.com/jerry0012000/CS6140ProjectTask1_Data_Synthesize
-   In Analyze GRU&LSTM folder, original data is from Hongzhen Xu folder, please check BASE_DIR variable.



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