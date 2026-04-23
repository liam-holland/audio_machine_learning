import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import os
    import glob
    import librosa
    import numpy
    import sys
    import pandas
    import re
    import pedalboard as pb
    import pathlib
    import sklearn
    import marimo as mo
    import shutil
    import torch

    return librosa, numpy, os, pandas, pathlib, pb, shutil, sklearn


@app.cell
def _(os):
    # Get a list of the files on my drive
    file_list = os.listdir(r"C:\Users\Liam Holland\Documents\GitHub\audio_machine_learning\Summative Assignment 2\final_work\Medley-solos-DB")
    return (file_list,)


@app.cell
def _(file_list, pandas):
    # Convert into a dataframe so I can add a file_path column
    file_path_df = pandas.DataFrame(data=file_list)
    file_path_df = file_path_df.rename(columns={0:"file_path"})
    file_path_df["file_path"] = "./Medley-solos-DB/" + file_path_df["file_path"]

    # Extract the unique identifer so I can join on it
    file_path_df["uuid4"] = file_path_df["file_path"].str.extract(r"_.*_(.*).wav")
    return (file_path_df,)


@app.cell
def _(file_path_df, pandas):
    # Read in the metadata and join it to the file path
    metadata_df = pandas.read_csv("Medley-solos-DB_metadata.csv")

    # Remove unneccary columns
    metadata_df = metadata_df.drop(["subset", "instrument_id","song_id"], axis=1, errors='ignore')

    # Remove electric guitar
    metadata_df = metadata_df[metadata_df["instrument"] != "distorted electric guitar"]

    metadata_df['uuid4'] = metadata_df['uuid4'].astype(str)
    file_path_df['uuid4'] = file_path_df['uuid4'].astype(str)

    joined_df = pandas.merge(left=metadata_df, right=file_path_df , how="left", on="uuid4")
    return (joined_df,)


@app.cell
def _(joined_df, pandas, sklearn):
    # Split the model into train test and val

    train, temp = sklearn.model_selection.train_test_split(joined_df, train_size = 0.7 , test_size = 0.3, shuffle=True, stratify = joined_df["instrument"] )

    test, val = sklearn.model_selection.train_test_split(temp, train_size = 0.5 , test_size = 0.5, shuffle=True, stratify = temp["instrument"] )

    # Limit the result to exactly 140 of each instrument
    train_filtered = train.groupby("instrument").head(140)
    train_filtered["test_train_val"] = "train"


    # Limit the result to exactly 30 of each instrument. 30 for validation and 30 for testing
    test_filtered = test.groupby("instrument").head(30)
    test_filtered["test_train_val"] = "test"
    val_filtered = val.groupby("instrument").head(30)
    val_filtered["test_train_val"] = "val"

    filtered_df = pandas.concat([train_filtered, test_filtered, val_filtered]).reset_index(drop=True)
    return (filtered_df,)


@app.cell
def _(filtered_df):
    filtered_df
    return


@app.cell
def _(pathlib, shutil):
    def copy_files_from_df(df, path_column, destination_folder):

        # 1. Ensure the destination exists
        dest_path = pathlib.Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        # 2. Iterate and copy
        for original_path_str in df[path_column]:
            path_str = original_path_str
            source = pathlib.Path(path_str)

            if source.exists():
                # This copies to destination/filename.wav
                new_source_name = "dry_" + source.name
                shutil.copy2(source, dest_path / new_source_name)
            else:
                print(f"Skipping: File not found at {source}")

    return (copy_files_from_df,)


@app.cell
def _(copy_files_from_df, filtered_df):
    copy_files_from_df( filtered_df, "file_path", f"./{len(filtered_df)}_raw_samples/")
    return


@app.cell
def _(numpy, pathlib, pb, shutil):
    # Set up the random effect settings.
    # Each effect gets low / mid / high bins.
    EFFECT_BIN_LABELS = ("low", "mid", "high")
    EFFECT_SAMPLING_SEED = 42
    effect_rng = numpy.random.default_rng(EFFECT_SAMPLING_SEED)


    EFFECT_GENERATION_CONFIG = {
        "reverb": {
            "primary_parameter": "room_size",
            "bins": {
                "low": (0.35, 0.50),
                "mid": (0.50, 0.65),
                "high": (0.65, 0.80),
            },
            "secondary_ranges": {
                "damping": (0.35, 0.65),
                "mix": (0.30, 0.50),
            },
        },
        "delay": {
            "primary_parameter": "delay_seconds",
            "bins": {
                "low": (0.18, 0.27),
                "mid": (0.27, 0.36),
                "high": (0.36, 0.45),
            },
            "secondary_ranges": {
                "feedback": (0.20, 0.40),
                "mix": (0.25, 0.45),
            },
        },
        "distortion": {
            "primary_parameter": "drive_db",
            "bins": {
                "low": (12.0, 16.0),
                "mid": (16.0, 20.0),
                "high": (20.0, 24.0),
            },
            "secondary_ranges": {
                "mix": (0.30, 0.50),
            },
        },
        "chorus": {
            "primary_parameter": "depth",
            "bins": {
                "low": (0.20, 0.267),
                "mid": (0.267, 0.333),
                "high": (0.333, 0.40),
            },
            "secondary_ranges": {
                "rate_hz": (0.8, 1.5),
                "centre_delay_ms": (6.0, 9.0),
                "feedback": (0.05, 0.18),
                "mix": (0.25, 0.45),
            },
        },
    }


    def sample_effect_settings(effect_name, bin_label, rng=None):
        if effect_name not in EFFECT_GENERATION_CONFIG:
            raise ValueError(f"Unsupported effect: {effect_name}")

        effect_config = EFFECT_GENERATION_CONFIG[effect_name]
        primary_parameter = effect_config["primary_parameter"]

        if bin_label not in effect_config["bins"]:
            available_bins = ", ".join(EFFECT_BIN_LABELS)
            raise ValueError(f"Unsupported bin '{bin_label}' for {effect_name}. Available: {available_bins}")

        rng = effect_rng if rng is None else rng
        primary_low, primary_high = effect_config["bins"][bin_label]
        effect_settings = {
            primary_parameter: float(rng.uniform(primary_low, primary_high)),
        }

        for parameter_name, (low, high) in effect_config["secondary_ranges"].items():
            effect_settings[parameter_name] = float(rng.uniform(low, high))

        return effect_settings


    def get_primary_parameter_name(effect_name):
        if effect_name not in EFFECT_GENERATION_CONFIG:
            raise ValueError(f"Unsupported effect: {effect_name}")

        return EFFECT_GENERATION_CONFIG[effect_name]["primary_parameter"]


    def build_effect(effect_name, effect_settings):
        if effect_name == "reverb":
            return pb.Reverb(
                room_size=effect_settings["room_size"],
                damping=effect_settings["damping"],
                width=1.0,
                freeze_mode=0.0,
            )

        if effect_name == "delay":
            return pb.Delay(
                delay_seconds=effect_settings["delay_seconds"],
                feedback=effect_settings["feedback"],
            )

        if effect_name == "distortion":
            return pb.Distortion(drive_db=effect_settings["drive_db"])

        if effect_name == "chorus":
            return pb.Chorus(
                rate_hz=effect_settings["rate_hz"],
                depth=effect_settings["depth"],
                centre_delay_ms=effect_settings["centre_delay_ms"],
                feedback=effect_settings["feedback"],
            )

        raise ValueError(f"Unsupported effect: {effect_name}")


    def applyEffect(effect_name, df_row, folder_output, effect_settings=None, bin_label=None):
        # file_path = pathlib.Path("/Users/gleborlov/Downloads") / df_row["file_path"].lstrip("./")
        file_path = pathlib.Path(df_row["file_path"])
        output_dir = pathlib.Path(folder_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        is_dry = effect_name.lower() == "dry"
        effect_settings = {} if effect_settings is None else dict(effect_settings)
        wet = 1.0 if is_dry else float(effect_settings["mix"])
        path_output = str(output_dir / f"{effect_name}_{df_row['uuid4']}.wav")

        try:
            if is_dry:
                shutil.copy2(file_path, path_output)
            else:
                effect = build_effect(effect_name, effect_settings)

                if hasattr(effect, "dry_level"):
                    effect.dry_level = 0.0
                    effect.wet_level = 1.0

                if hasattr(effect, "mix"):
                    effect.mix = 1.0

                board = pb.Pedalboard([effect])
                board.reset()

                with pb.io.AudioFile(str(file_path)) as f:
                    with pb.io.AudioFile(path_output, "w", f.samplerate, f.num_channels) as o:
                        while f.tell() < f.frames:
                            chunk = f.read(f.samplerate)
                            effected = board(chunk, f.samplerate, reset=False)
                            mixed_audio = wet * effected + (1 - wet) * chunk
                            o.write(mixed_audio)

            new_row = df_row.copy()
            new_row["new_file_path"] = path_output
            new_row["effect_applied"] = effect_name
            new_row["wet_dry"] = f"{wet:.4f}"
            if is_dry:
                new_row["parameter"] = "none"
                new_row["parameter_value"] = "{}"
            else:
                primary_parameter = get_primary_parameter_name(effect_name)
                new_row["parameter"] = f"{primary_parameter}:{bin_label}"
                new_row["parameter_value"] = str(effect_settings)
            return [new_row]

        except Exception as e:
            print(f"Error processing {file_path} with {effect_name}: {e}")
            return []

    return EFFECT_BIN_LABELS, applyEffect, sample_effect_settings


@app.cell
def _(df_row, pathlib):
    file_path = pathlib.Path("/Users/gleborlov/Downloads") / df_row["file_path"].lstrip("./")
    return


@app.cell
def _(
    EFFECT_BIN_LABELS,
    applyEffect,
    filtered_df,
    pandas,
    sample_effect_settings,
):
    # Make the dry and effected files.
    # Keep the split labels while doing it.
    rows_list = []
    effect_folders = {
        "reverb": "reverb/",
        "delay": "delay/",
        "distortion": "distortion/",
        "chorus": "chorus/",
    }
    split_col = "test_train_val"

    for split_name in ["train", "val", "test"]:
        split_samples = filtered_df[filtered_df[split_col] == split_name].reset_index(drop=True)

        for _, sample in split_samples.iterrows():
            rows_list.extend(applyEffect("dry", sample, "dry/"))

        for effect_name, folder_output in effect_folders.items():
            for sample_idx, (_, sample) in enumerate(split_samples.iterrows()):
                bin_label = EFFECT_BIN_LABELS[sample_idx % len(EFFECT_BIN_LABELS)]
                effect_settings = sample_effect_settings(effect_name, bin_label)
                rows_list.extend(
                    applyEffect(
                        effect_name,
                        sample,
                        folder_output,
                        effect_settings=effect_settings,
                        bin_label=bin_label,
                    )
                )

    new_df = pandas.DataFrame(rows_list).reset_index(drop=True)
    new_df["split"] = new_df["test_train_val"]
    new_df_split = new_df.copy()
    new_df.to_csv("train_test_val_mapping.csv", index=False)
    new_df = new_df.drop(columns=["test_train_val"])
    new_df
    return (new_df_split,)


@app.cell
def _(new_df_split):
    # Generation already assigned split labels with balanced effect bins.
    # Use that split directly instead of re-splitting here.
    new_df_split_1 = new_df_split.copy()
    new_df_split_1.to_csv('new_df_split.csv', index=False)
    return (new_df_split_1,)


@app.cell
def _(new_df_split_1):
    # `new_df_split` already contains the authoritative split column.
    new_df_split_1[['uuid4', 'instrument', 'effect_applied', 'split']].head()
    return


@app.cell
def _(librosa, numpy):
    # Small helpers so the feature code is less repetitive.
    def summarize_feature(values, feature_name, stats=("mean", "std")):
        stat_functions = {
            "mean": numpy.mean,
            "std": numpy.std,
            "min": numpy.min,
            "max": numpy.max,
        }

        flattened = numpy.asarray(values).ravel()
        summary = {}

        for stat_name in stats:
            if stat_name not in stat_functions:
                available_stats = ", ".join(sorted(stat_functions))
                raise ValueError(f"Unknown stat '{stat_name}'. Available stats: {available_stats}")

            summary[f"{feature_name}_{stat_name}"] = float(stat_functions[stat_name](flattened))

        return summary


    def safe_ratio(numerator, denominator, eps=1e-8):
        return float(numerator / max(float(denominator), eps))


    def extract_baseline_features_from_signal(y, sr, n_mfcc=13, stats=("mean", "std")):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        feature_blocks = {
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
            "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
            "zcr": librosa.feature.zero_crossing_rate(y),
        }

        features = {}

        for i in range(n_mfcc):
            features.update(summarize_feature(mfcc[i], f"mfcc_{i+1}", stats=stats))

        for feature_name, values in feature_blocks.items():
            features.update(summarize_feature(values, feature_name, stats=stats))

        return features


    def extract_delta_mfcc_features_from_signal(y, sr, n_mfcc=13, stats=("mean", "std")):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)

        features = {}

        for i in range(n_mfcc):
            features.update(summarize_feature(delta_mfcc[i], f"delta_mfcc_{i+1}", stats=stats))

        return features


    def extract_rms_features_from_signal(y, sr, stats=("mean", "std")):
        rms = librosa.feature.rms(y=y)
        return summarize_feature(rms, "rms_energy", stats=stats)


    def extract_spectral_bandwidth_features_from_signal(y, sr, stats=("mean", "std")):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        return summarize_feature(spectral_bandwidth, "spectral_bandwidth", stats=stats)


    def extract_spectral_flatness_features_from_signal(y, sr, stats=("mean", "std")):
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        return summarize_feature(spectral_flatness, "spectral_flatness", stats=stats)


    def extract_energy_shape_features_from_signal(y, sr, split_fraction=0.5):
        split_idx = int(len(y) * split_fraction)
        split_idx = min(max(split_idx, 1), max(len(y) - 1, 1))

        head = y[:split_idx]
        tail = y[split_idx:]
        rms = librosa.feature.rms(y=y).ravel()

        head_energy = numpy.mean(numpy.square(head)) if len(head) else 0.0
        tail_energy = numpy.mean(numpy.square(tail)) if len(tail) else 0.0

        return {
            "tail_head_energy_ratio": safe_ratio(tail_energy, head_energy),
            "rms_range": float(numpy.max(rms) - numpy.min(rms)),
        }


    def extract_rms_autocorrelation_features_from_signal(y, sr, max_lag=32):
        rms = librosa.feature.rms(y=y).ravel()
        centered_rms = rms - numpy.mean(rms)

        if len(centered_rms) < 2 or numpy.allclose(centered_rms, 0.0):
            return {
                "rms_autocorr_mean": 0.0,
                "rms_autocorr_std": 0.0,
                "rms_autocorr_peak": 0.0,
                "rms_autocorr_peak_lag": 0.0,
            }

        autocorr = numpy.correlate(centered_rms, centered_rms, mode="full")
        autocorr = autocorr[len(centered_rms) - 1:]
        autocorr = autocorr / max(float(autocorr[0]), 1e-8)

        positive_lags = autocorr[1:min(len(autocorr), max_lag + 1)]
        if len(positive_lags) == 0:
            positive_lags = numpy.array([0.0])

        peak_offset = int(numpy.argmax(positive_lags))

        return {
            "rms_autocorr_mean": float(numpy.mean(positive_lags)),
            "rms_autocorr_std": float(numpy.std(positive_lags)),
            "rms_autocorr_peak": float(positive_lags[peak_offset]),
            "rms_autocorr_peak_lag": float(peak_offset + 1),
        }


    def extract_centroid_modulation_features_from_signal(y, sr, stats=("mean", "std")):
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
        centroid_modulation = numpy.abs(numpy.diff(spectral_centroid))

        if len(centroid_modulation) == 0:
            centroid_modulation = numpy.array([0.0])

        return summarize_feature(centroid_modulation, "centroid_modulation", stats=stats)


    def extract_spectral_flux_features_from_signal(y, sr, stats=("mean", "std")):
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        return summarize_feature(spectral_flux, "spectral_flux", stats=stats)


    # All the feature extractors are collected here.
    FEATURE_EXTRACTORS = {
        "baseline": extract_baseline_features_from_signal,
        "delta_mfcc": extract_delta_mfcc_features_from_signal,
        "energy_shape": extract_energy_shape_features_from_signal,
        "rms_energy": extract_rms_features_from_signal,
        "rms_autocorrelation": extract_rms_autocorrelation_features_from_signal,
        "spectral_bandwidth": extract_spectral_bandwidth_features_from_signal,
        "spectral_flatness": extract_spectral_flatness_features_from_signal,
        "centroid_modulation": extract_centroid_modulation_features_from_signal,
        "spectral_flux": extract_spectral_flux_features_from_signal,
    }


    def extract_audio_features(audio_path, feature_sets=("baseline",), sr=22050, mono=True, feature_kwargs=None):
        if isinstance(feature_sets, str):
            feature_sets = [feature_sets]

        feature_kwargs = feature_kwargs or {}
        y, sr = librosa.load(audio_path, sr=sr, mono=mono)

        features = {}

        for feature_set in feature_sets:
            if feature_set not in FEATURE_EXTRACTORS:
                available_feature_sets = ", ".join(sorted(FEATURE_EXTRACTORS))
                raise ValueError(
                    f"Unknown feature set '{feature_set}'. Available feature sets: {available_feature_sets}"
                )

            extractor_kwargs = feature_kwargs.get(feature_set, {})
            feature_values = FEATURE_EXTRACTORS[feature_set](y=y, sr=sr, **extractor_kwargs)

            duplicate_keys = set(features).intersection(feature_values)
            if duplicate_keys:
                duplicate_keys_str = ", ".join(sorted(duplicate_keys))
                raise ValueError(f"Duplicate feature names found: {duplicate_keys_str}")

            features.update(feature_values)

        return features


    def extract_baseline_features(audio_path, sr=22050, n_mfcc=13, stats=("mean", "std"), mono=True):
        return extract_audio_features(
            audio_path=audio_path,
            feature_sets=("baseline",),
            sr=sr,
            mono=mono,
            feature_kwargs={"baseline": {"n_mfcc": n_mfcc, "stats": stats}},
        )

    return (extract_audio_features,)


@app.cell
def _(extract_audio_features, pandas, pathlib):
    # Loop through the audio files and build one big feature table.
    def build_feature_dataframe(
        new_df,
        audio_col="new_file_path",
        feature_sets=("baseline",),
        sr=22050,
        mono=True,
        feature_kwargs=None,
        verbose=True,
        raise_on_error=False,
    ):
        rows = []
        total_samples = len(new_df)

        for i, row in new_df.iterrows():
            if verbose:
                print(f"Processing sample {i} / {total_samples}")

            audio_path = pathlib.Path(row[audio_col])

            try:
                feat_dict = extract_audio_features(
                    audio_path=audio_path,
                    feature_sets=feature_sets,
                    sr=sr,
                    mono=mono,
                    feature_kwargs=feature_kwargs,
                )
                out_row = row.to_dict()
                out_row.update(feat_dict)
                rows.append(out_row)
            except Exception as e:
                print(f"Failed on {audio_path}: {e}")
                if raise_on_error:
                    raise

        return pandas.DataFrame(rows)

    return (build_feature_dataframe,)


@app.cell
def _(build_feature_dataframe, new_df_split_1):
    # Pick which features to turn on here.
    FEATURE_CONFIG = {'feature_sets': ('baseline', 'energy_shape', 'rms_energy', 'rms_autocorrelation', 'spectral_bandwidth', 'spectral_flatness', 'centroid_modulation', 'spectral_flux'), 'feature_kwargs': {'baseline': {'n_mfcc': 13, 'stats': ('mean', 'std')}, 'energy_shape': {'split_fraction': 0.5}, 'rms_energy': {'stats': ('mean', 'std')}, 'rms_autocorrelation': {'max_lag': 32}, 'spectral_bandwidth': {'stats': ('mean', 'std')}, 'spectral_flatness': {'stats': ('mean', 'std')}, 'centroid_modulation': {'stats': ('mean', 'std')}, 'spectral_flux': {'stats': ('mean', 'std')}}}
    feature_df = build_feature_dataframe(new_df_split_1, audio_col='new_file_path', **FEATURE_CONFIG)  # "delta_mfcc",  # "delta_mfcc": {  #     "n_mfcc": 13,  #     "stats": ("mean", "std"),  # },
    return (feature_df,)


@app.cell
def _(feature_df):
    # Pull the effect label back out of the generated file path.
    feature_df["effect"] = (
        feature_df["new_file_path"]
        .astype(str)
        .str.lstrip("./")
        .str.split("/")
        .str[0]
    )

    feature_df
    return


@app.cell
def _(feature_df):
    # Save the features and keep them available for the merged experiment section below.
    feature_df.to_csv("feature_df.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
