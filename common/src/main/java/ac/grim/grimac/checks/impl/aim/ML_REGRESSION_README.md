# ML Regression Model - Documentation

## Overview

The ML system has been completely rewritten from **classification** (cheat/legit) to **regression** (next-tick prediction).

### Key Concept

Instead of classifying whether a player is cheating, the model now **predicts the next yaw/pitch movement** based on the last 40 ticks.

**Why this approach?**
- **Legit players**: Movements are chaotic, noisy, with micro-corrections → High prediction error
- **Aimbot users**: Movements are smooth, predictable, pattern-based → Low prediction error

**Lower prediction error = More likely to be cheating**

---

## Architecture Changes

### Input Data
- **Sequence**: 40 ticks
- **Features per tick**: 8
  - `deltaYaw`, `deltaPitch` (rotation deltas)
  - `yawAccel`, `pitchAccel` (acceleration)
  - `yawJerk`, `pitchJerk` (jerk - change in acceleration)
  - `gcdYawError`, `gcdPitchError` (GCD rounding errors)
- **Shape**: `[40][8]`

### Output Data
- **Output**: 2 values
  - `[nextDeltaYaw, nextDeltaPitch]`
- **Activation**: TANH (bounded output)
- **Loss**: MSE (Mean Squared Error)

### Neural Network
```
BatchNormalization (nIn=8)
    ↓
LSTM 256 (TANH) + Dropout(0.3)
    ↓
LSTM 128 (TANH) + Dropout(0.3)
    ↓
Dense 64 (ReLU) + Dropout(0.4)
    ↓
Dense 32 (ReLU) + Dropout(0.4)
    ↓
Output 2 (TANH, MSE)
```

---

## Training Process

### Data Preparation

1. **Load CSV files** from `plugins/GrimAC/ml_datasets/`
2. **Filter for legit data only** (critical!)
   - Only LEGIT data is used for training
   - CHEAT data is loaded but filtered out
3. **Create sequences**: Sliding window of 40 ticks → predict 1 tick ahead

### Normalization

Computed on training data:
- `mean` and `std` for each of the 8 features
- Applied: `(x - mean) / std`, clamped to `[-5, 5]`
- **Saved** in model file for inference

### Training

- **Optimizer**: Adam (lr=0.001)
- **Early Stopping**: 15 epochs patience
- **Learning Rate Decay**: 0.5x every 20 epochs
- **Validation**: 80/20 train/validation split
- **Metrics**:
  - **MSE** (Mean Squared Error)
  - **MAE** (Mean Absolute Error)
- **No classification metrics** (accuracy, precision, recall, F1 - removed!)

### Export

Model weights are exported to JSON for use in Grim:
```json
{
  "version": "2.0",
  "lstm1_W": [...],
  "lstm1_RW": [...],
  "lstm1_b": [...],
  "lstm2_W": [...],
  "lstm2_RW": [...],
  "lstm2_b": [...],
  "dense1_W": [...],
  "dense1_b": [...],
  "dense2_W": [...],
  "dense2_b": [...],
  "out_W": [...],
  "out_b": [...],
  "featureMeans": [...],
  "featureStds": [...],
  "validationMSE": 0.001234,
  "validationMAE": 0.023456
}
```

Saved to: `plugins/GrimAC/ml_models/aim_model_weights.json`

---

## Inference (Detection)

### Process

1. **Collect 40 ticks** of player rotation data
2. **Normalize** using saved means/stds
3. **Predict** next yaw/pitch delta
4. **Calculate error**:
   ```
   predictionError = |predictedYaw - realYaw| + |predictedPitch - realPitch|
   ```

### Detection Thresholds

- **High error** (≥ 0.5°): Legit (chaotic movement)
- **Low error** (≤ 0.05°): Suspicious (highly predictable)
- **Critical error** (≤ 0.02°): Very likely aimbot

### Violation Level (VL)

- **Increases**: When prediction error is very low
- **Decreases**: When prediction error is high (legit behavior)
- **Flag**: At VL ≥ 20

---

## Commands

### Data Collection

```bash
# Start recording for yourself
/grimAiOn

# Stop and save as LEGIT_<player>_<timestamp>.csv
/grimAiOff

# Mark current recording as CHEAT
/shame ban <player>
```

### Global Collection

```bash
# Start recording from ALL players (except those with grim.ml.exempt)
/grimAiGlobal start <collection_id>

# Stop and create ZIP archive
/grimAiGlobal stop
```

### Training

```bash
# Train regression model from all CSV files
/grimAiTrain

# This will:
# 1. Load all CSV files
# 2. Filter for LEGIT data only
# 3. Create training examples (40-tick windows)
# 4. Train the neural network
# 5. Save model (DL4J format)
# 6. Export weights to JSON
```

### Status

```bash
# Show ML status for player (or yourself)
/grimAiStatus [player]

# List all datasets
/grimAiList
```

---

## File Locations

```
plugins/GrimAC/
├── ml_datasets/               # CSV files (LEGIT_*, CHEAT_*)
│   ├── LEGIT_player1_20240101-120000.csv
│   ├── CHEAT_player2_20240101-130000.csv
│   └── GLOBAL_collection_20240101.zip
│
└── ml_models/
    ├── aim_model.dat          # DL4J model (for training)
    ├── aim_model.dat.stats    # Normalization stats
    └── aim_model_weights.json # Exported weights (for Grim inference)
```

---

## CSV Format

Each CSV file contains raw tick data:

```csv
delta_yaw,delta_pitch,accel_yaw,accel_pitch,jerk_yaw,jerk_pitch,gcd_error_yaw,gcd_error_pitch
0.5234,-0.1234,0.0123,-0.0045,0.0023,0.0012,0.0001,0.0002
-0.3456,0.2345,-0.0087,0.0035,-0.0056,0.0010,0.0000,0.0001
...
```

**Note**: No cheat/legit label in CSV - determined by filename prefix:
- `LEGIT_*.csv` → isCheat = 0.0
- `CHEAT_*.csv` → isCheat = 1.0

---

## Important Notes

### Training Data Quality

1. **Collect A LOT of legit data** (at least 5000+ ticks)
2. **From multiple players** with different playstyles
3. **Different scenarios**:
   - PvP combat
   - Building
   - Exploring
   - Sitting still
   - Quick turns
   - Tracking targets

4. **DO NOT train on cheat data!** The model learns human motorics only.

### Detection Behavior

- **False positives**: Can happen if legit player is very consistent
- **False negatives**: Sophisticated aimbots with human-like randomness
- **Tuning**: Adjust thresholds in `ImprovedAimML.java`:
  ```java
  private static final double PREDICT_ERROR_HIGH = 0.5;
  private static final double PREDICT_ERROR_LOW = 0.05;
  private static final double PREDICT_ERROR_CRITICAL = 0.02;
  ```

### Performance

- **Model size**: ~500K parameters (small, fast inference)
- **Prediction time**: < 1ms per sequence
- **Memory**: Minimal (no DL4J needed at runtime if using JSON weights)

---

## Migration from Old Model

If upgrading from classification model:

1. **Old CSV files are compatible** (same format)
2. **Old model files won't work** (different architecture)
3. **Retrain with new approach**:
   - Collect more LEGIT data
   - Run `/grimAiTrain`
4. **New detection logic**:
   - Old: `probability > threshold` → cheat
   - New: `predictionError < threshold` → cheat

---

## Troubleshooting

### "Not enough legit ticks for training!"

Solution: Collect more LEGIT data. The model only trains on legit data.

### "Model dimension mismatch!"

Solution: Retrain the model with `/grimAiTrain`. The model is now expecting [40][8] input.

### "High false positive rate!"

Possible causes:
1. Not enough variety in training data
2. Model overfitting (too complex for data size)
3. Thresholds too strict

Solutions:
1. Collect more diverse legit data
2. Reduce model size (fewer LSTM neurons)
3. Increase thresholds

### "Aimbot not detected!"

Possible causes:
1. Aimbot has human-like randomness
2. Training data is too clean/predictable
3. Thresholds too lenient

Solutions:
1. Lower thresholds
2. Add more natural variation to training data
3. Consider adding additional features

---

## Future Improvements

1. **Multiple models**: Different models for different scenarios (PvP, building, etc.)
2. **Online learning**: Update model continuously with new legit data
3. **Attention mechanism**: Better temporal pattern detection
4. **Ensemble models**: Combine multiple models for better accuracy
5. **Feature engineering**: Add more features (mouse movement timestamps, click patterns, etc.)

---

## References

- DL4J Documentation: https://deeplearning4j.knowledgebase/
- LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Aimbot Detection Literature: Various academic papers on cheat detection

---

**Version**: 2.0 - Regression Edition
**Date**: 2024
**Authors**: GrimAC ML Team
