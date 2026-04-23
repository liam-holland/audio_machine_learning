# AML Project README

This file is for the role C mainly

Main idea:
- `data_generation.ipynb` is for making the dataset and feature table
- `model_experiments.ipynb` is for training models, trying settings, and writing down results

## Project Files

- [data_generation.ipynb]
- [model_experiments.ipynb]
- [feature_df.csv]
- [new_df_split.csv]
- [train_test_val_mapping.csv]

Generated audio folders:
- [dry]
- [chorus]
- [delay]
- [distortion]
- [reverb]

## Before Running Anything

You need:
- the original Medley dataset
- `Medley-solos-DB_metadata.csv`
- a Python environment with the packages listed below

If dataset or metadata locations differ on your machine, update the notebook cell that points to them.

If you are running notebooks from VS Code or Jupyter, make sure the kernel uses that venv or another environment with:
- `numpy`
- `pandas`
- `librosa`
- `scikit-learn`
- `torch`
- `matplotlib`
- `pedalboard`

## Normal Workflow

Run things in this order:

1. Open [data_generation.ipynb]
2. Run all cells there if you need to rebuild the generated audio or feature table
3. Check that these files were updated:
   - [feature_df.csv]
   - [new_df_split.csv]
4. Open [model_experiments.ipynb]
5. Run experiments there
6. Record results, findings, and any parameter changes you made

If you only want to try new models on the same existing data/features, you usually only need to run:
- [model_experiments.ipynb](

## For The Next Person

The next person should focus on experiments and analysis, not data regeneration.

Default approach:
- use [model_experiments.ipynb]
- change config dictionaries, not the training pipeline
- rerun only the section related to the model you changed
- record `val_macro_f1`, `test_macro_f1`, and confusion matrices

Main knobs to change:
- tabular models: `MODEL_NAME` and `MODEL_KWARGS`
- mel / hybrid CNN: `MEL_CNN_CONFIG`

Good experiment loop:
1. change one config block
2. rerun that experiment section
3. note exact settings changed
4. compare validation first, then test
5. keep the confusion matrix for error analysis

Only go back to [data_generation.ipynb] if you are changing:
- effect generation settings
- split generation
- extracted feature sets in `FEATURE_CONFIG`

If you do not change those, do not rebuild the dataset.

## What `data_generation.ipynb` Does

This notebook does 3 jobs:

1. Load and filter the raw Medley files
2. Generate effected audio with Pedalboard
3. Build a tabular feature dataframe

### Split Logic

The notebook creates `train / val / test` splits first.

Then for each split it:
- copies a dry version
- makes 1 version for each effect:
  - `reverb`
  - `delay`
  - `distortion`
  - `chorus`

Important:
- it cycles `low -> mid -> high` bins separately inside each split
- this is done so each split gets rough coverage of effect strength bins

### Effect Parameters

The effect setup is in `EFFECT_GENERATION_CONFIG` inside [data_generation.ipynb]
Current primary binning:
- `reverb` uses `room_size`
- `delay` uses `delay_seconds`
- `distortion` uses `drive_db`
- `chorus` uses `depth`

If you want to change how strong the generated effects are, change:
- the bin ranges
- the secondary random ranges

### Feature Extraction

The feature setup is controlled by `FEATURE_CONFIG` in [data_generation.ipynb]

Current active feature groups:
- `baseline`
- `energy_shape`
- `rms_energy`
- `rms_autocorrelation`
- `spectral_bandwidth`
- `spectral_flatness`
- `centroid_modulation`
- `spectral_flux`

What they roughly mean:
- `baseline`: MFCCs + centroid + rolloff + zero crossing rate
- `energy_shape`: tail/head energy ratio and RMS range
- `rms_energy`: RMS mean/std
- `rms_autocorrelation`: repeating pattern info from the RMS envelope
- `spectral_bandwidth`: how spread the spectrum is
- `spectral_flatness`: how noise-like vs tone-like it is
- `centroid_modulation`: how much centroid changes over time
- `spectral_flux`: how much the spectrum changes over time

If you want to turn features on or off:
- go to `FEATURE_CONFIG`
- edit `feature_sets`
- edit `feature_kwargs` if needed

Example:

```python
FEATURE_CONFIG = {
    "feature_sets": (
        "baseline",
        "spectral_flatness",
    ),
    "feature_kwargs": {
        "baseline": {
            "n_mfcc": 13,
            "stats": ("mean", "std"),
        },
        "spectral_flatness": {
            "stats": ("mean", "std"),
        },
    },
}
```

### Outputs From `data_generation.ipynb`

After running it, the main saved outputs are:
- [feature_df.csv]
- [new_df_split.csv]
- [train_test_val_mapping.csv]

Use `new_df_split.csv` as the main audio-level split file.

## What `model_experiments.ipynb` Does

This notebook has 2 experiment tracks:

1. Tabular models on the extracted features
2. Mel spectrogram CNN / hybrid CNN on the audio

Its training section should be treated as the main place to run experiments.
Use it by changing config dicts, not by editing the training functions unless there is a real bug.

It loads:
- [feature_df.csv]
- [new_df_split.csv]

If `new_df_split.csv` is missing, it falls back to:
- [train_test_val_mapping.csv]

## Tabular Models

The tabular pipeline does:

1. load `feature_df.csv`
2. recover the effect label
3. encode labels
4. split into train / val / test
5. standardize features using train only
6. train the chosen sklearn model

### Available Tabular Models

Current supported names:
- `logistic_regression`
- `mlp`
- `random_forest`
- `svm`

Also supported aliases:
- `logistic`
- `logreg`
- `nn`
- `rf`
- `randomforest`
- `svc`

### Where To Change the Tabular Model

In [model_experiments.ipynb], edit:

```python
MODEL_NAME = "mlp"
MODEL_KWARGS = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-4,
    "max_iter": 500,
    "early_stopping": True,
}
```

Practical rule:
- change `MODEL_NAME` when switching model family
- change `MODEL_KWARGS` when tuning that model
- leave the rest of the tabular pipeline alone

### Meaning of Main Tabular Parameters

#### Logistic Regression

Common ones:
- `max_iter`: max solver iterations
- `class_weight`: use `"balanced"` if class imbalance becomes a problem
- `C`: regularization strength control

Example:

```python
MODEL_NAME = "logistic_regression"
MODEL_KWARGS = {
    "max_iter": 2000,
    "class_weight": "balanced",
}
```

#### SVM

Common ones:
- `kernel`: usually `"rbf"` or `"linear"`
- `C`: penalty strength
- `gamma`: RBF kernel width, common value is `"scale"`
- `class_weight`: can use `"balanced"`

Example:

```python
MODEL_NAME = "svm"
MODEL_KWARGS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
}
```

#### Random Forest

Common ones:
- `n_estimators`: number of trees
- `max_depth`: max tree depth
- `min_samples_leaf`: minimum leaf size
- `class_weight`: can use `"balanced"`

Example:

```python
MODEL_NAME = "random_forest"
MODEL_KWARGS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 5,
}
```

#### MLP

Common ones:
- `hidden_layer_sizes`: network size
- `alpha`: regularization
- `learning_rate_init`: starting learning rate
- `max_iter`: max iterations
- `early_stopping`: stop when validation stops improving

Example:

```python
MODEL_NAME = "mlp"
MODEL_KWARGS = {
    "hidden_layer_sizes": (128, 64),
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": True,
}
```

### Tabular Outputs To Look At

After running the tabular section, check:
- `history_df`
- `train_metrics`
- `val_metrics`
- `test_metrics`
- `train_cm_df`
- `val_cm_df`
- `test_cm_df`

Main scores:
- `accuracy`
- `macro_f1`

Use `val_macro_f1` as the main model comparison score unless the team decides otherwise.

## Mel CNN / Hybrid CNN

This section trains directly on mel spectrograms instead of the feature table.

Basic flow:

1. load audio
2. resample
3. pad/truncate to fixed length
4. compute mel spectrogram
5. convert to dB
6. normalize
7. train CNN

### Available Mel Model Variants

Current supported values:
- `cnn`
- `hybrid`

Meaning:
- `cnn`: mel spectrogram only
- `hybrid`: mel spectrogram + small auxiliary handcrafted feature vector

### Hybrid Auxiliary Features

Current available aux feature sets:
- `rms_stats`
- `spectral_flatness`
- `spectral_rolloff`
- `hf_energy_ratio`

What they mean:
- `rms_stats`: RMS mean, RMS std, RMS range
- `spectral_flatness`: flatness mean/std
- `spectral_rolloff`: rolloff mean/std
- `hf_energy_ratio`: how much energy sits in the high-frequency region

### Where To Change the Mel Model

Edit `MEL_CNN_CONFIG` in [model_experiments.ipynb]

Current example:

```python
MEL_CNN_CONFIG = {
    "sample_rate": 22050,
    "fixed_duration": 3.0,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "power": 2.0,
    "normalize": "per_sample",
    "model_variant": "hybrid",
    "aux_feature_sets": ("rms_stats", "spectral_flatness"),
    "aux_clip_value": 5.0,
    "batch_size": 32,
    "num_epochs": 200,
    "validation_every_n_epochs": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler_name": "reduce_on_plateau",
    "lr_scheduler_monitor": "val_macro_f1",
    "best_model_monitor": "val_macro_f1",
    "lr_scheduler_kwargs": {
        "factor": 0.5,
        "patience": 2,
        "min_lr": 1e-6,
    },
    "dropout": 0.3,
    "cache_spectrograms": True,
}
```

Practical rule:
- use `model_variant` to switch between `cnn` and `hybrid`
- use `aux_feature_sets` to test different hybrid inputs
- use the scheduler and validation settings only when tuning training behaviour
- leave the mel training code alone unless something is broken

### Meaning of Main Mel Parameters

- `sample_rate`: audio resample rate
- `fixed_duration`: clip length in seconds after pad/truncate
- `n_mels`: number of mel bins
- `n_fft`: FFT window size
- `hop_length`: hop between frames
- `power`: power spectrogram setting before dB conversion
- `normalize`: mel normalization mode, currently usually `"per_sample"`
- `model_variant`: `"cnn"` or `"hybrid"`
- `aux_feature_sets`: which aux handcrafted features to concatenate in hybrid mode
- `aux_clip_value`: clip standardized aux features to control outliers
- `batch_size`: samples per gradient step
- `num_epochs`: number of epochs
- `validation_every_n_epochs`: how often validation runs
- `learning_rate`: optimizer learning rate
- `weight_decay`: optimizer regularization
- `dropout`: dropout in classifier head
- `cache_spectrograms`: keep computed spectrograms in memory during a run

Scheduler-related:
- `lr_scheduler_name`: currently use `None` or `"reduce_on_plateau"`
- `lr_scheduler_monitor`: metric used to decide when LR changes
- `best_model_monitor`: metric used to decide which checkpoint is saved as best
- `lr_scheduler_kwargs["factor"]`: how much to reduce LR by
- `lr_scheduler_kwargs["patience"]`: how many validation checks to wait
- `lr_scheduler_kwargs["min_lr"]`: lower LR limit

Important:
- `validation_every_n_epochs` changes how often the scheduler sees validation
- if you validate every epoch, a small scheduler patience can reduce LR too early

### Good First Mel Experiments

#### Plain CNN

```python
MEL_CNN_CONFIG["model_variant"] = "cnn"
```

#### Hybrid CNN

```python
MEL_CNN_CONFIG["model_variant"] = "hybrid"
MEL_CNN_CONFIG["aux_feature_sets"] = ("rms_stats", "spectral_flatness")
```

#### Try More Aux Features

```python
MEL_CNN_CONFIG["aux_feature_sets"] = (
    "rms_stats",
    "spectral_flatness",
    "spectral_rolloff",
    "hf_energy_ratio",
)
```

#### Validate More Often

```python
MEL_CNN_CONFIG["validation_every_n_epochs"] = 1
MEL_CNN_CONFIG["lr_scheduler_kwargs"] = {
    "factor": 0.5,
    "patience": 8,
    "min_lr": 1e-6,
}
```

## Mel Outputs To Look At

After running the mel section, check:
- `mel_cnn_history_df`
- `mel_cnn_train_metrics`
- `mel_cnn_val_metrics`
- `mel_cnn_test_metrics`
- `mel_cnn_train_cm_df`
- `mel_cnn_val_cm_df`
- `mel_cnn_test_cm_df`
- `mel_cnn_aux_diagnostics_df`

If using hybrid:
- `mel_cnn_aux_feature_names`
- `mel_cnn_aux_feature_shape`
- `mel_cnn_hybrid_tsne_df`

`mel_cnn_aux_diagnostics_df` is useful for spotting:
- NaNs
- infs
- very large raw ranges
- too much clipping

Things to watch:
- `non_finite_count`
- `abs_standardized_max`
- `clip_rate`

## What To Record In Your Results

For every experiment, write down:

1. notebook used
2. model type
3. exact parameters changed
4. feature sets used
5. train / val / test accuracy
6. train / val / test macro F1
7. confusion matrix summary
8. any obvious failure pattern

Good examples of findings:
- dry is often confused with chorus
- distortion is easiest class
- adding spectral flatness helps dry vs effect
- random forest overfits train but does not help val/test
- validating every epoch made LR drop too early

## Suggested Experiment Table

Use something like this in your notes or report:

| Experiment | Input Type | Model | Main Settings | Val Macro F1 | Test Macro F1 | Notes |
|---|---|---|---|---:|---:|---|
| E1 | tabular | logistic regression | baseline defaults |  |  |  |
| E2 | tabular | random forest | 200 trees |  |  |  |
| E3 | tabular | mlp | `(64, 32)` |  |  |  |
| E4 | mel | cnn | default mel config |  |  |  |
| E5 | mel | hybrid | `rms_stats + spectral_flatness` |  |  |  |
| E6 | mel | hybrid | add `spectral_rolloff` |  |  |  |

## Common Problems

### Missing CSV files

If `model_experiments.ipynb` says artifacts are missing:
- rerun [data_generation.ipynb]

### Feature columns include labels or split strings

The experiment notebook already filters non-numeric columns.
If you change the dataframe structure, double-check `feature_cols`.

### Mel training looks stuck

Check:
- learning rate
- scheduler patience
- validation frequency
- aux diagnostics if using hybrid

### Hybrid t-SNE errors

t-SNE is only for the hybrid model because it uses the combined vector before classification.
If using plain CNN, switch `model_variant` back to `"hybrid"` first.
