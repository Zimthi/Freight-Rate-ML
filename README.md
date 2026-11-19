# Freight Rate Prediction

Gradient-boosted model (log-rate, median loss) with time-based validation.

## Run
```bash
python -m pip install -r requirements.txt
python train.py        # CV metrics -> metrics.json, writes validation_predictions.csv + december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```
Data goes in `data/` (`train-test.csv`, `validation.csv`, `december-chart-inputs.csv`, `validation-predictions-template.csv`).

## Layout
- `src/features.py` cleaning + feature engineering
- `train.py` validation, final fit, predictions
- `report.pdf` approach, validation, December chart
