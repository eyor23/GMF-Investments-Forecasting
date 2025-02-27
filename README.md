## Overview

`eda.ipynb` is a Jupyter Notebook designed to demonstrate the application of the `eda_analysis.py` module for exploratory data analysis (EDA). The notebook provides a structured, step-by-step workflow to analyze a dataset and derive meaningful insights.

## Features

This notebook includes:

- Loading and exploring the dataset.
- Generating summary statistics for numerical columns.
- Visualizing distributions of numerical and categorical features.
- Performing correlation analysis to identify relationships between numerical features.
- Detecting missing values and outliers in the dataset.

## Installation and Requirements

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python libraries:
  - `pandas`
  - `matplotlib`
  - `seaborn`

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required libraries:
   ```bash
   pip install pandas matplotlib seaborn
   ```

## Usage

### Running the Notebook

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook eda.ipynb
   ```
2. Follow the structured workflow in the notebook to analyze your dataset step-by-step.

### Example Workflow

1. **Importing the Module**

   ```python
   import eda_analysis as eda
   ```

2. **Loading the Dataset**

   ```python
   df = eda.load_data('data.csv')
   ```

3. **Overview of the Dataset**

   ```python
   eda.overview_data(data)
   ```

4. **Summary Statistics**

   ```python
   eda.summary_statistics(data)
   ```

5. **Visualize Numerical Distributions**

   ```python
   eda.visualize_numerical_distribution(data, ['Amount', 'Value'])
   ```

6. **Detect Outliers**

   ```python
   eda.detect_outliers(data, ['Amount', 'Value'])
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- `pandas` for data manipulation
- `matplotlib` and `seaborn` for visualization

