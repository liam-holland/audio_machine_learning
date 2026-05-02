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

    return librosa, numpy, pandas, pathlib


@app.cell
def _(pandas):
    # Get a list of the files on my drive
    new_df_split_1 = pandas.read_csv(filepath_or_buffer=r"./new_df_split.csv")
    return (new_df_split_1,)


@app.cell
def _(numpy):
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

    return (summarize_feature,)


@app.cell
def _(librosa, summarize_feature):
    def extract_rms_features_from_signal(y, sr, stats=("mean", "std")):
        rms = librosa.feature.rms(y=y)
        return summarize_feature(rms, "rms_energy", stats=stats)

    return (extract_rms_features_from_signal,)


@app.cell
def _(librosa, summarize_feature):
    def extract_spectral_flatness_features_from_signal(y, sr, stats=("mean", "std")):
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        return summarize_feature(spectral_flatness, "spectral_flatness", stats=stats)

    return (extract_spectral_flatness_features_from_signal,)


@app.cell
def _(
    extract_rms_features_from_signal,
    extract_spectral_flatness_features_from_signal,
):
    # All the feature extractors are collected here.
    FEATURE_EXTRACTORS = {
        "rms_energy": extract_rms_features_from_signal,
        "spectral_flatness": extract_spectral_flatness_features_from_signal,
    }
    return (FEATURE_EXTRACTORS,)


@app.cell
def _(FEATURE_EXTRACTORS, librosa):
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
    FEATURE_CONFIG = {'feature_sets': ('rms_energy', 'spectral_flatness'), 'feature_kwargs': {'rms_energy': {'stats': ('mean', 'std')}, 'spectral_flatness': {'stats': ('mean', 'std')}}}

    feature_df = build_feature_dataframe(new_df_split_1, audio_col='new_file_path', **FEATURE_CONFIG) 
    return


@app.cell
def _():
    # feature_df.to_csv("feature_df.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
