# Reproducibility Report

## Environment Specifications
- **Python Version**: 3.12.10
- **OS Platform**: Windows-11-10.0.26200-SP0
- **Random Seed**: 42

## Execution Order to Reproduce Results from Scratch
1. **Environment Setup**:
   ```bash
   pip install -r environment/requirements.txt
   ```
2. **Train and Serialize Checkpoints**:
   ```bash
   python code/training/train.py
   ```
   This will align raw TIFFs, extract features, fit models, serialize imputers/ensembles to `models/`, and output the baseline `outputs/submission.csv`.
3. **Run Inference from Checkpoints**:
   ```bash
   python code/inference/predict.py
   ```
   This loads the pre-trained picklings, runs inference, and writes `submission_regenerated.csv`.

## Verification Procedure
Run the following script to verify that your regenerated predictions match our final submission exactly:
```python
import pandas as pd
import numpy as np
sub1 = pd.read_csv('outputs/submission.csv')
sub2 = pd.read_csv('submission_regenerated.csv')
assert sub1.shape == sub2.shape, "Shape mismatch"
assert np.allclose(sub1.iloc[:, 1:].values, sub2.iloc[:, 1:].values, atol=1e-5), "Predictions mismatch"
print("PASS: Reproducibility verified successfully.")
```
