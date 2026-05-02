import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import torch
    import pathlib
    import pandas
    import sklearn
    from torch.utils.data import DataLoader, Dataset
    import json
    import hashlib
    import numpy
    from torch import nn
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    return (
        DataLoader,
        Dataset,
        accuracy_score,
        confusion_matrix,
        f1_score,
        hashlib,
        json,
        nn,
        numpy,
        pandas,
        pathlib,
        plt,
        sklearn,
        sns,
        torch,
    )


@app.cell
def _():
    LABEL_COL = "effect_applied"
    return (LABEL_COL,)


@app.cell
def _(HybridMelSpectrogramCNN, MelSpectrogramDataset, torch):
    def load_and_initialize_model(test_df, label_encoder, weights_path="best_hybrid_cnn.pth"):
        # 1. Configuration (Must match your training settings)
        dataset_kwargs = {
            "audio_col": "new_file_path",
            "label_col": "label",
            "sample_rate": 22050,
            "fixed_duration": 3.0,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 256,
            "power": 2.0,
            "normalize": "per_sample",
            "aux_feature_sets": ("rms_stats", "spectral_flatness"),
            "cache_dir": "mel_cache_v1",
            "use_disk_cache": True,
        }

        # 2. Use the test_df to infer shapes
        dataset = MelSpectrogramDataset(test_df, **dataset_kwargs)
        mel, aux, _, instr = dataset[0]

        num_classes = len(label_encoder.classes_)
        aux_feature_dim = len(aux)

        # 3. Build Architecture
        model = HybridMelSpectrogramCNN(
            num_classes=num_classes,
            aux_feature_dim=aux_feature_dim,
        )

        # 4. Initialise LazyLinear layers with a dummy forward pass
        model.eval()
        with torch.no_grad():
            _ = model(mel.unsqueeze(0), aux.unsqueeze(0))

        # 5. Load Weights
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return model, dataset_kwargs

    return (load_and_initialize_model,)


@app.cell
def _():
    return


@app.cell
def _(DataLoader, MelSpectrogramDataset, torch):
    def get_model_predictions(model, test_df, dataset_kwargs):
        test_dataset = MelSpectrogramDataset(test_df, **dataset_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_true = []
        all_instr = []

        model.eval()
        with torch.no_grad():
            for xb, auxb, yb, instr_ids in test_loader:
                logits = model(xb, auxb)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(yb.cpu().numpy())
                all_instr.extend(instr_ids)

        return all_true, all_preds, all_instr

    return (get_model_predictions,)


@app.cell
def _(accuracy_score, confusion_matrix, f1_score, numpy, pandas, plt, sns):
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """
        Plots high-res Heatmap and prints Accuracy/F1 metrics.
        Uses numpy and pandas (full names).
        """
        # --- 1. Calculate and Print Accuracy Info ---
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        print("\n" + "="*40)
        print("HYBRID CNN - TEST RESULTS")
        print("="*40)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Macro F1: {f1:.4f}")
        print("-" * 40)

        # --- 2. Visual Setup ---
        CELL_TEXT_SIZE = 14
        LABEL_SIZE = 16
        TICK_SIZE = 14
        TITLE_SIZE = 22

        cm = confusion_matrix(y_true, y_pred)
        # Replaced pd with pandas
        cm_df = pandas.DataFrame(cm, index=class_names, columns=class_names)
        cm_perc = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)

        # Replaced np with numpy
        annot_labels = numpy.array([
            [f"{perc:.1%}\n({int(count)})" for count, perc in zip(row_counts, row_percs)]
            for row_counts, row_percs in zip(cm_df.values, cm_perc.values)
        ])

        plt.figure(figsize=(12, 9))
        sns.set_theme(style="white")

        ax = sns.heatmap(
            cm_perc, 
            annot=annot_labels, 
            fmt="", 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            annot_kws={"size": CELL_TEXT_SIZE, "weight": "bold"}
        )

        title_text = "Hybrid CNN - Test Confusion Matrix\n(Mel + Physical Features)"
        plt.title(title_text, pad=30, fontsize=TITLE_SIZE, fontweight='bold')
        plt.ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold')

        # Fix rotations
        plt.xticks(rotation=45, ha='right', fontsize=TICK_SIZE)
        plt.yticks(rotation=0, fontsize=TICK_SIZE) 

        plt.tight_layout()
        plt.savefig(f'./images/{title_text.replace(chr(10), " ")}.png', bbox_inches='tight', dpi=300)

        return plt.gcf()

    return (plot_confusion_matrix,)


@app.cell
def _(pandas, pathlib):
    # Reuse in-memory artifacts from the generation cells when available.
    # Fall back to the exported CSVs if this section is run on its own.
    FEATURE_DF_PATH = pathlib.Path("feature_df.csv")
    AUDIO_SPLIT_CANDIDATES = (
        pathlib.Path("new_df_split.csv"),
        pathlib.Path("train_test_val_mapping.csv"),
    )

    def load_notebook_artifact(csv_path):
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {csv_path}. Run data_generation.ipynb through the export cells first."
            )

        dataframe = pandas.read_csv(csv_path)
        unnamed_cols = [c for c in dataframe.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            dataframe = dataframe.drop(columns=unnamed_cols)
        return dataframe

    if "feature_df" in globals():
        feature_df = feature_df.copy()
    else:
        feature_df = load_notebook_artifact(FEATURE_DF_PATH)

    if "new_df_split" in globals():
        new_df_split = new_df_split.copy()
    else:
        audio_split_path = next((path for path in AUDIO_SPLIT_CANDIDATES if path.exists()), None)
        if audio_split_path is None:
            raise FileNotFoundError(
                "Missing audio split artifact. Run data_generation.ipynb so it exports new_df_split.csv or train_test_val_mapping.csv."
            )
        new_df_split = load_notebook_artifact(audio_split_path)

    return feature_df, new_df_split


@app.cell
def _(LABEL_COL, feature_df, pandas, sklearn):
    label_encoder = sklearn.preprocessing.LabelEncoder()


    # Prefer the generation-time split and keep only numeric feature columns.
    split_col = "split" if "split" in feature_df.columns else "test_train_val"

    excluded_cols = {
        "instrument",
        "uuid4",
        "file_path",
        "new_file_path",
        "wet_dry",
        "parameter",
        "parameter_value",
        "effect_applied",
        "effect",
        "label",
        "split",
        "test_train_val",
    }


    feature_df["effect"] = feature_df[LABEL_COL]

    if feature_df["effect"].isna().any():
        raise ValueError("Missing values in effect_applied — labels are broken.")

    feature_cols = [
        c for c in feature_df.columns
        if c not in excluded_cols and pandas.api.types.is_numeric_dtype(feature_df[c])
    ]





    feature_df["label"] = label_encoder.fit_transform(feature_df["effect"])
    split_col = "split" if "split" in feature_df.columns else "test_train_val"
    test_df = feature_df[feature_df[split_col] == "test"].copy()
    return label_encoder, test_df


@app.cell
def _(
    Dataset,
    HYBRID_AUX_FEATURE_BUILDERS,
    PRECOMPUTED_AUX_FEATURE_SPECS,
    get_precomputed_aux_source_columns,
    hashlib,
    json,
    librosa,
    numpy,
    pandas,
    pathlib,
    torch,
):
    def _make_cache_spec(sample_rate, fixed_duration, n_mels, n_fft, hop_length, power, normalize, aux_feature_sets):
        return {
            "sample_rate": sample_rate,
            "fixed_duration": fixed_duration,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "power": power,
            "normalize": normalize,
            "aux_feature_sets": list(normalize_aux_feature_sets(aux_feature_sets)),
        }

    def _make_row_cache_key(row, cache_spec, audio_col="new_file_path"):
        base = {
            "audio_path": str(row[audio_col]),
            "uuid4": str(row["uuid4"]) if "uuid4" in row.index else "",
            "spec": cache_spec,
        }
        payload = json.dumps(base, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


    class MelSpectrogramDataset(Dataset):
        def __init__(
            self,
            dataframe,
            audio_col="new_file_path",
            label_col="label",
            sample_rate=22050,
            fixed_duration=3.0,
            n_mels=128,
            n_fft=1024,
            hop_length=256,
            power=2.0,
            normalize="per_sample",
            aux_feature_sets=("rms_stats", "spectral_flatness"),
            aux_feature_stats=None,
            aux_clip_value=5.0,
            cache_spectrograms=True,
            cache_dir="mel_cache_v1",
            use_disk_cache=True,
        ):
            self.dataframe = dataframe.reset_index(drop=True)
            self.audio_col = audio_col
            self.label_col = label_col
            self.sample_rate = sample_rate
            self.fixed_duration = fixed_duration
            self.n_mels = n_mels
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.power = power
            self.normalize = normalize
            self.aux_feature_sets = normalize_aux_feature_sets(aux_feature_sets)
            self.aux_feature_stats = aux_feature_stats
            self.aux_clip_value = aux_clip_value
            self.cache_spectrograms = cache_spectrograms
            self.use_disk_cache = use_disk_cache
            self.cache_dir = pathlib.Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.num_samples = int(sample_rate * fixed_duration)
            self._cache = {} if cache_spectrograms else None
            self.aux_feature_names = None
            self.precomputed_aux_source_columns = get_precomputed_aux_source_columns(self.aux_feature_sets)
            self.cache_spec = _make_cache_spec(
                sample_rate=sample_rate,
                fixed_duration=fixed_duration,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                power=power,
                normalize=normalize,
                aux_feature_sets=self.aux_feature_sets,
            )

        def __len__(self):
            return len(self.dataframe)

        def _load_audio(self, audio_path):
            y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            if len(y) < self.num_samples:
                y = numpy.pad(y, (0, self.num_samples - len(y)))
            else:
                y = y[:self.num_samples]
            return y

        def _normalize_mel(self, mel_db):
            if self.normalize in [None, False, "none"]:
                return mel_db
            if self.normalize == "per_sample":
                mean = numpy.mean(mel_db)
                std = numpy.std(mel_db)
                return (mel_db - mean) / (std + 1e-8)
            if self.normalize == "minmax":
                min_val = numpy.min(mel_db)
                max_val = numpy.max(mel_db)
                return (mel_db - min_val) / (max_val - min_val + 1e-8)
            raise ValueError("normalize must be one of None, 'per_sample', or 'minmax'.")

        def _sanitize_aux_vector(self, aux_vector):
            aux_vector = numpy.asarray(aux_vector, dtype=numpy.float32)
            return numpy.nan_to_num(aux_vector, nan=0.0, posinf=0.0, neginf=0.0)

        def _get_cache_path(self, row):
            cache_key = _make_row_cache_key(row, self.cache_spec, audio_col=self.audio_col)
            return self.cache_dir / f"{cache_key}.npz"

        def _get_precomputed_aux_features(self, row):
            if row is None:
                return None
            if not self.aux_feature_sets:
                return numpy.zeros(0, dtype=numpy.float32), []
            if not self.precomputed_aux_source_columns:
                return None
            if not all(source_col in row.index and pandas.notna(row[source_col]) for source_col in self.precomputed_aux_source_columns):
                return None

            feature_names = []
            feature_values = []
            for feature_set in self.aux_feature_sets:
                specs = PRECOMPUTED_AUX_FEATURE_SPECS.get(feature_set)
                if not specs:
                    return None
                for feature_name, source_col in specs:
                    feature_names.append(feature_name)
                    feature_values.append(float(row[source_col]))
            aux_vector = self._sanitize_aux_vector(feature_values)
            return aux_vector, feature_names

        def _compute_aux_features(self, y=None, row=None):
            if not self.aux_feature_sets:
                return numpy.zeros(0, dtype=numpy.float32), []

            precomputed = self._get_precomputed_aux_features(row)
            if precomputed is not None:
                return precomputed
            if y is None:
                raise ValueError("Audio waveform is required when precomputed aux features are unavailable.")

            feature_values = {}
            for feature_set in self.aux_feature_sets:
                if feature_set not in HYBRID_AUX_FEATURE_BUILDERS:
                    available = ", ".join(sorted(HYBRID_AUX_FEATURE_BUILDERS))
                    raise ValueError(f"Unknown aux_feature_set '{feature_set}'. Available: {available}")
                feature_values.upandasate(HYBRID_AUX_FEATURE_BUILDERS[feature_set](y=y, sr=self.sample_rate))

            feature_names = list(feature_values.keys())
            aux_vector = self._sanitize_aux_vector([feature_values[name] for name in feature_names])
            return aux_vector, feature_names

        def _normalize_aux_vector(self, aux_vector):
            if self.aux_feature_stats is not None and len(aux_vector):
                aux_vector = (aux_vector - self.aux_feature_stats["mean"]) / self.aux_feature_stats["std"]
            if self.aux_clip_value is not None and len(aux_vector):
                aux_vector = numpy.clip(aux_vector, -self.aux_clip_value, self.aux_clip_value)
            return aux_vector.astype(numpy.float32)

        def _compute_and_optionally_cache_disk_item(self, row):
            y = self._load_audio(row[self.audio_col])

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=self.power,
            )
            mel_db = librosa.power_to_db(mel, ref=numpy.max)
            mel_db = self._normalize_mel(mel_db).astype(numpy.float32)

            aux_vector, feature_names = self._compute_aux_features(y=y, row=row)
            aux_vector = self._normalize_aux_vector(aux_vector)

            if self.use_disk_cache:
                cache_path = self._get_cache_path(row)
                if not cache_path.exists():
                    numpy.savez_compressed(
                        cache_path,
                        mel=mel_db.astype(numpy.float32),
                        aux=aux_vector.astype(numpy.float32),
                        aux_names=numpy.asarray(feature_names, dtype=object),
                    )

            return mel_db, aux_vector, feature_names

        def _compute_item(self, index):
            row = self.dataframe.iloc[index]

            if self.use_disk_cache:
                cache_path = self._get_cache_path(row)
                if cache_path.exists():
                    cached = numpy.load(cache_path, allow_pickle=True)
                    mel_db = cached["mel"].astype(numpy.float32)
                    aux_vector = cached["aux"].astype(numpy.float32)
                    feature_names = list(cached["aux_names"])
                else:
                    mel_db, aux_vector, feature_names = self._compute_and_optionally_cache_disk_item(row)
            else:
                mel_db, aux_vector, feature_names = self._compute_and_optionally_cache_disk_item(row)

            mel_tensor = torch.from_numpy(mel_db).unsqueeze(0)
            aux_tensor = torch.from_numpy(aux_vector)

            if self.aux_feature_names is None:
                self.aux_feature_names = feature_names

            instr_id = row['instrument']

            label_tensor = torch.tensor(int(row[self.label_col]), dtype=torch.long)
            return mel_tensor, aux_tensor, label_tensor, instr_id

        def __getitem__(self, index):
            if self._cache is not None and index in self._cache:
                return self._cache[index]

            item = self._compute_item(index)
            if self._cache is not None:
                self._cache[index] = item
            return item

    return (MelSpectrogramDataset,)


@app.function
def normalize_aux_feature_sets(aux_feature_sets):
    if aux_feature_sets is None:
        return ()
    if isinstance(aux_feature_sets, str):
        return (aux_feature_sets,)
    return tuple(aux_feature_sets)


@app.cell
def _(PRECOMPUTED_AUX_FEATURE_SPECS):
    def get_precomputed_aux_source_columns(aux_feature_sets):
        columns = []
        for feature_set in normalize_aux_feature_sets(aux_feature_sets):
            columns.extend(source_col for _, source_col in PRECOMPUTED_AUX_FEATURE_SPECS.get(feature_set, ()))
        return tuple(dict.fromkeys(columns))

    return (get_precomputed_aux_source_columns,)


@app.cell
def _():
    PRECOMPUTED_AUX_FEATURE_SPECS = {
        "rms_stats": (
            ("rms_mean", "rms_energy_mean"),
            ("rms_std", "rms_energy_std"),
        ),
        "spectral_flatness": (
            ("spectral_flatness_mean", "spectral_flatness_mean"),
            ("spectral_flatness_std", "spectral_flatness_std"),
        ),
        "spectral_rolloff": (
            ("spectral_rolloff_mean", "spectral_rolloff_mean"),
            ("spectral_rolloff_std", "spectral_rolloff_std"),
        ),
        "hf_energy_ratio": (
            ("hf_energy_ratio_mean", "hf_energy_ratio_mean"),
            ("hf_energy_ratio_std", "hf_energy_ratio_std"),
        ),
    }
    return (PRECOMPUTED_AUX_FEATURE_SPECS,)


@app.cell
def _(MelSpectrogramEncoder, nn, torch):
    class HybridMelSpectrogramCNN(nn.Module):
        def __init__(self, num_classes, aux_feature_dim, dropout=0.3):
            super().__init__()
            self.encoder = MelSpectrogramEncoder(dropout=dropout)
            self.classifier = nn.Sequential(
                nn.Linear(48 + aux_feature_dim, 48),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(48, num_classes),
            )

        def encode(self, x):
            return self.encoder(x)

        def encode_with_aux(self, x, aux_features):
            embedding = self.encode(x)
            combined = torch.cat([embedding, aux_features], dim=1)
            return embedding, combined

        def forward(self, x, aux_features=None):
            if aux_features is None:
                raise ValueError("HybridMelSpectrogramCNN requires aux_features.")
            embedding = self.encoder(x)
            combined = torch.cat([embedding, aux_features], dim=1)
            return self.classifier(combined)

    return (HybridMelSpectrogramCNN,)


@app.cell
def _(nn):
    class MelSpectrogramEncoder(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(48),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = self.features(x)
            return self.projection(x)

    return (MelSpectrogramEncoder,)


@app.cell
def _(
    get_model_predictions,
    label_encoder,
    load_and_initialize_model,
    test_df,
):
    model_final, dataset_kwargs = load_and_initialize_model(test_df, label_encoder, weights_path="best_hybrid_cnn.pth")
    class_names_final = list(label_encoder.classes_)

    # 1. Generate Prediction Data
    y_true, y_pred, instruments = get_model_predictions(model_final, test_df, dataset_kwargs)
    return class_names_final, y_pred, y_true


@app.cell
def _(class_names_final, plot_confusion_matrix, y_pred, y_true):
    cm_fig = plot_confusion_matrix(y_true, y_pred, class_names_final)
    return (cm_fig,)


@app.cell
def _(cm_fig):
    cm_fig

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
