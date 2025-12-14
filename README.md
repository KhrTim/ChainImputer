# ChainImputer

A neural network-based iterative imputation method using cumulative features for handling missing values in datasets.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

ChainImputer is a novel approach to missing value imputation that addresses the limitations of conventional methods. Rather than applying rough initial guesses across all missing entries, ChainImputer incrementally builds a set of reliable features during neural network training. This progressive strategy avoids reliance on unreliable preliminary data and demonstrates superior prediction accuracy.

## Key Features

- **Iterative Feature Construction**: Progressively builds cumulative feature sets during training
- **Asymmetric Strategy**: Incorporates newly imputed features incrementally rather than all at once
- **Reliable Feature Leverage**: Uses only carefully validated features at each iteration
- **Superior Performance**: Tested on 25 public datasets with significant improvements over conventional approaches
- **Neural Network-Based**: Leverages deep learning for adaptive imputation patterns

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KhrTim/ChainImputer.git
cd ChainImputer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate chainimputer
```

## Usage

### Basic Example

```python
from chainimputer import ChainImputer
import pandas as pd

# Load your data with missing values
data = pd.read_csv('your_data.csv')

# Initialize ChainImputer
imputer = ChainImputer(
    max_iterations=10,
    learning_rate=0.001,
    hidden_layers=[64, 32]
)

# Fit and transform
imputed_data = imputer.fit_transform(data)

print("Imputation complete!")
```

### Running Experiments

```bash
# Run experiments on public datasets
python new_experiment.py --dataset <dataset_name> --iterations 10

# Update or retrain models
python update.py --config config.yml
```

### Command-line Interface

```bash
# Impute a CSV file
python -m chainimputer impute --input data.csv --output imputed_data.csv

# Evaluate on a specific dataset
python -m chainimputer evaluate --dataset dataset_name --metrics all
```

## Methodology

ChainImputer employs a unique iterative approach:

1. **Initial Phase**: Identifies complete features and highly reliable partial features
2. **Iterative Imputation**:
   - Trains neural network on cumulative reliable feature set
   - Imputes next most reliable feature based on learned patterns
   - Adds newly imputed feature to cumulative set
3. **Progressive Refinement**: Each iteration expands the reliable feature set
4. **Final Imputation**: Completes all missing values with progressively learned patterns

This asymmetric strategy ensures that each imputation step builds on previously validated features rather than unreliable initial guesses.

## Experimental Results

Tested on 25 public datasets including:
- UCI Machine Learning Repository datasets
- Kaggle competition datasets
- Healthcare and financial datasets

**Key Findings:**
- Superior prediction accuracy compared to MICE, KNN, and MissForest
- Robust performance across varying missing data rates (10%-50%)
- Effective handling of both MAR (Missing At Random) and MNAR (Missing Not At Random) patterns

## Project Structure

```
ChainImputer/
├── preprocessing/          # Data preprocessing modules
├── models/                 # Neural network architectures
├── experiments/            # Experimental scripts and configs
├── new_experiment.py       # Main experiment runner
├── update.py              # Model update utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## Citation

If you use ChainImputer in your research, please cite our paper:

```bibtex
@article{seo2025chainimputer,
  title={ChainImputer: A Neural Network-Based Iterative Imputation Method Using Cumulative Features},
  author={Seo, Wangduk and Khairulov, Timur and Baek, Hye-Jin and Lee, Jaesung},
  journal={Symmetry},
  volume={17},
  number={6},
  pages={869},
  year={2025},
  publisher={MDPI},
  doi={10.3390/sym17060869}
}
```

**Paper Link**: [Symmetry Journal](https://scholar.google.com/citations?view_op=view_citation&hl=ru&user=-XrW5PAAAAAJ&citation_for_view=-XrW5PAAAAAJ:9yKSN-GCB0IC)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- PyTorch >= 1.9.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Wangduk Seo**
- **Timur Khairulov** - [GitHub](https://github.com/KhrTim)
- **Hye-Jin Baek**
- **Jaesung Lee**

## Acknowledgments

- Thanks to the UCI Machine Learning Repository for providing benchmark datasets
- Inspired by iterative imputation methods and neural network architectures
- Special thanks to all contributors and reviewers

## Contact

For questions or feedback, please open an issue on GitHub or contact the authors through the paper correspondence.

---

**Note**: This is a research project. For production use, please thoroughly validate on your specific datasets and use cases.
