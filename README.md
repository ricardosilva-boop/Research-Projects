# Research-Projects
My projects aim to apply machine learning methods to solve problems related to Mining Engineering.
# Project Goal for 3D CNN
This project aims to estimate Molybdenum (Mo) concentration in a 3D geological model using a 3D Convolutional Neural Network (CNN). The goal is to build a robust model that can predict Mo concentrations based on spatial coordinates (X, Y, Z) and associated Copper (Cu) values and then evaluate its performance using various statistical metrics and visualizations.

Notebook Steps
The following steps were executed in the notebook to achieve the project goal:
1.	Import Libraries: Essential libraries for data manipulation (pandas, numpy), spatial interpolation (scipy), machine learning (sklearn, tensor- flow) and plotting (matplotlib) were imported.
2.	Load Data: The dataset Cu_Mo_processed.csv was loaded, containing pre-processed spatial coordinates (X, Y, Z) and concentrations of Copper (Cu) and Molybdenum (Mo). An additional dataset Cu_Mo.csv was loaded for evaluation.
3.	Define 3D Grid: A regular 3D grid was defined across the spatial extent of the data. The grid cell dimensions were set to CELL_X=25, CELL_Y=25, CELL_Z=25, with a PADDING of 0.2 to cover the full data range.
4.	Normalize and Interpolate: Mo and Cu concentration values were nor- malized using MinMaxScaler. These scaled values were then interpolated onto the defined 3D grid using the griddata function with the 'nearest' method to create a volume array. This volume represents the spatial distribution of normalized Mo and Cu.
5.	Create Input Patches for CNN: For each data point (drill hole location), a 3x3x3x2 patch was extracted from the volume array. These patches, representing local 3D neighborhoods of Mo and Cu concentrations, served as input (X_train) for the 3D CNN model. The corresponding Molybdenum values were used as labels (y_train).
6.	Define and Train 3D CNN Model: A sequential 3D CNN model was constructed using tensorflow.keras. The model includes Conv3D layers, MaxPooling3D, GlobalAveragePooling3D, and Dense layers. It was compiled with the Adam optimizer and Mean Squared Error (mse) loss function. The model was trained for 50 epochs with a batch size of 16 and a 20% validation split.
7.	Estimate Blocks with Batch Inference: A set of prediction points (grid_pts) was generated for all blocks within the convex hull of the origi- nal data points. For each valid block, a 3x3x3x2 patch was extracted from the volume. The trained CNN model then performed batch inference on these patches to predict Mo concentrations (mo_pred_opt).
8.	Save Block Model: The estimated Mo concentrations (both log- transformed and exponentially transformed) were compiled into a DataFrame (df_blocks_cnn) along with their spatial coordinates and saved to modelo_blocos_CNN_2_2.csv.
9.	Evaluate Model Performance: The estimated Mo concentrations from the block model were compared against actual Mo values from the df_furos dataset by associating drill holes with their nearest estimated blocks.
10.	Calculate Metrics: Key evaluation metrics including Root Mean Squared Error (RMSE), Mean Error (ME), Correlation (Corr), and R-squared (R2) were calculated for both log-transformed and original Mo values. Additionally, skewness and kurtosis of the residuals were computed.
11.	Generate Plots: Histograms of residuals (log and original) with superim- posed normal distribution curves, and scatter plots of real vs. estimated Mo values (log and original), were generated to visually assess model performance and residual distribution.

Key Evaluation Metrics Summary
The model’s performance was evaluated on both log-transformed and original scales:
Log Scale: * RMSE: 0.2573 * ME: 0.0966 * Correlation (r): 0.8836 * R2: 0.7447 * Skewness: 0.4656 * Kurtosis: 5.8790
Original Scale (Bias-Corrected): * RMSE: 0.0294 * ME: 0.0042 * Correlation (r): 0.7831 * R2: 0.5912 * Skewness: 9.6105 * Kurtosis: 166.5242
The original scale metrics were calculated using a bias correction term (exp(Mo_est_log + 0.5 * Var_log)), where Var_log was approximated by the variance of Mo_est_CNN_log from the df_blocos due to the absence of specific variance estimates per block.

Generated Plots
•	Residual Histograms (log and original scale): These plots show the distribution of the difference between real and estimated Mo values. For the log scale, the residuals tend to be closer to a normal distribution, while on the original scale, the distribution is highly skewed, indicating challenges in predicting higher concentrations accurately and the need for log-transformation for modeling.
•	Scatterplots of Real vs. Estimated Values (log and original scale): These plots visualize the correlation between the actual and predicted Mo concentrations. The log scale plot shows a strong linear relationship, confirming the high correlation coefficient. The original scale plot shows more scatter, especially at higher concentrations, which is reflected in the lower correlation and R2 score compared to the log scale.
 
How to Reproduce all Cells Quickly
To reproduce the results, ensure you have the following CSV files in the same directory as the notebook: * Cu_Mo_processed.csv * Cu_Mo.csv * cnn_3d_var_log.csv (if using the variance from a previously saved file, otherwise df_blocos["Mo_est_CNN_log"].var( ) is used as a fallback).
Run all cells sequentially in the provided Jupyter Notebook or Colab environment. The modelo_blocos_CNN_2_2.csv file, containing the estimated block model, will be generated upon successful execution.


# Project Goal for Cokriging
This Project contains a Python script designed to perform Ordinary Cokriging (OKC). It estimates Molybdenum (Mo) grades using Copper (Cu) as a secondary covariable, generates a 3D block model, and performs a comprehensive statistical validation of the results.
How to Execute the Code
	Environment Setup: Ensure you have Python 3.8+ installed. The code is optimized for a Jupyter Notebook environment (.ipynb), but can also be run as a standard Python script (.py).
	Prepare the Data: Place your input dataset named Cu_Mo_processed.csv in the same directory as the script. The dataset must contain spatial coordinates (X, Y, Z) and log-transformed grades (Mo, Cu).
	Run the Code:
	If using Jupyter Notebook / Google Colab: Simply run the cells sequentially from top to bottom. The first cell will automatically install any missing dependencies (pyvista, scikit-learn).
	If using a standard Python IDE: Remove the !pip install line at the very top, manually install the requirements via your terminal (pip install pyvista scikit-learn numpy pandas matplotlib scipy tqdm), and run the script.
	Check Outputs: The script will output a block model file (modelo_blocos_deswik.csv) and save four high-resolution PDF plots in your working directory.
Code Structure (The 10 Cells)
Cell 1: Dependencies & Data Loading
Installs required libraries and imports the input drill hole data (Cu_Mo_processed.csv). It extracts the spatial coordinates (X, Y, Z) and the log-transformed Molybdenum and Copper values.
Cell 2: Variogram Parameters & Spatial Functions
Defines the spatial continuity of the deposit. It sets up the rotation matrix based on the azimuth (157.5°) and defines the primary (Spherical) and secondary (Gaussian) covariance models, alongside nugget effects, ranges, and the correlation coefficient (ρ=0.645).
Cell 3: Proportional OKC Estimator Function
Contains the core geostatistical engine (okc_point). For a given target point, it finds the nearest samples using a KDTree, builds the left-hand kriging matrix (covariances between samples) and the right-hand vector (covariances between samples and the target), and solves the linear system to calculate the weights, estimated value, and kriging variance.

Cell 4: 3D Grid Generation & Convex Hull Masking
Generates a 3D grid with 25 x 25 x 25 blocks covering the spatial extent of the data. To avoid estimating blocks in empty space (extrapolation), it uses a Delaunay triangulation and Convex Hull to mask and keep only the blocks that fall within the geometric limits of the actual drill holes.
Cell 5: Block Estimation & CSV Export
Iterates through all valid blocks (using a tqdm progress bar) and applies the Cokriging function. It applies a log-normal back-transformation formula to convert the estimates back to original units:〖Est〗_orig=exp⁡(〖Est〗_log+1/2 〖Var〗_log ). Finally, it exports the results to modelo_blocos_deswik.csv.
Cell 6: Validation Setup (Nearest Neighbor Matching)
Reloads the newly created block model and the original dataset. It uses a cKDTree to match every real drill hole to its closest estimated block centroid in 3D space, setting up the data for cross-validation.
Cell 7: Core Performance Metrics
Calculates standard validation metrics by comparing the real drill hole grades against the block model estimates. It computes the Root Mean Square Error (RMSE), Mean Error (ME), and Pearson Correlation for both the log-scale and the original scale.
Cell 8: Advanced Statistical Metrics
Calculates the Coefficient of Determination (R^2) to assess how well the block model explains the variance of the real data, alongside Skewness and Kurtosis to evaluate the shape and asymmetry of the error distributions.
Cell 9: Residual Distribution Analysis
Generates density histograms of the residuals (Real Grade - Estimated Grade) for both original and log scales. It overlays a theoretical Normal Distribution curve over the data and plots lines for the mean and standard deviation, saving the results as PDFs.
Cell 10: Real vs. Estimated Scatterplots
Plots the final scatterplots comparing the real grades against the estimated grades. It includes a red 1∶1 reference dashed line to visually evaluate overestimation or underestimation trends. These plots are also exported as PDFs.

