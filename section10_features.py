from collections import Counter
from pathlib import Path
import re

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kruskal, spearmanr


ROOT = Path("dataset")
DATA = ROOT / "dataset"
MERGE = ROOT / "build" / "MERGE_ECAC"
TRAIN_FILE = MERGE / "cleaned" / "metadata_clean_treino.csv"

METADATA_FEATURES = [
    "Relevance",
    "Duration",
    "Sample",
    "ActualYear",
    "Year_y",
    "LowestYear_y",
]

POSITIVE_WORDS = {
    "love", "loved", "loving", "happy", "happiness", "joy", "joyful",
    "smile", "smiles", "smiling", "hope", "hopeful", "light", "beautiful",
    "sweet", "good", "heart", "heaven", "angel", "peace", "free",
    "together", "kiss", "laugh", "laughter",
}
NEGATIVE_WORDS = {
    "sad", "sadness", "lonely", "alone", "pain", "hurt", "cry", "crying",
    "tears", "dark", "afraid", "fear", "hate", "angry", "anger", "broken",
    "goodbye", "lost", "regret", "cold", "death", "dying", "sorrow",
    "misery", "wrong", "shame", "trouble", "bad",
}
HIGH_ACTIVATION_WORDS = {
    "run", "running", "fight", "fire", "burn", "burning", "shout", "scream",
    "screaming", "wild", "rush", "jump", "dance", "dancing", "fast",
    "explode", "attack", "chase", "ride", "rage", "storm", "thunder",
    "sweat", "kill", "blood", "crash", "hard",
}
LOW_ACTIVATION_WORDS = {
    "calm", "quiet", "slow", "still", "sleep", "sleeping", "dream",
    "dreams", "rest", "gentle", "soft", "easy", "peace", "peaceful",
    "tired", "weary", "warm", "breathe", "silence", "hush", "wait",
    "lazy", "floating", "drift",
}

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


EXPECTED_AXES = {
    "Relevance": "Neither",
    "Duration": "Arousal",
    "Sample": "Neither",
    "ActualYear": "Neither",
    "Year_y": "Neither",
    "LowestYear_y": "Neither",
    "audio_major_minor_profile": "Valence",
    "audio_chroma_entropy_bits": "Neither",
    "audio_tempo_bpm": "Arousal",
    "audio_onsets_per_second": "Arousal",
    "audio_beat_interval_cv": "Arousal",
    "audio_rms_mean": "Arousal",
    "audio_spectral_centroid_hz": "Arousal",
    "audio_spectral_bandwidth_hz": "Arousal",
    "audio_spectral_rolloff85_hz": "Arousal",
    "audio_zero_crossing_rate": "Arousal",
    "lyrics_positive_rate": "Valence",
    "lyrics_negative_rate": "Valence",
    "lyrics_sentiment_balance": "Valence",
    "lyrics_high_arousal_rate": "Arousal",
    "lyrics_low_arousal_rate": "Arousal",
    "lyrics_type_token_ratio": "Neither",
    "lyrics_mean_token_length": "Neither",
    "lyrics_punctuation_per_100_words": "Neither",
    "lyrics_nonempty_line_count": "Neither",
    "lyrics_repeated_line_rate": "Neither",
    "cover_colorfulness": "Valence",
    "cover_saturation_mean": "Arousal",
    "cover_brightness_mean": "Arousal",
    "cover_grayscale_std": "Neither",
    "cover_grayscale_entropy_bits": "Neither",
    "cover_edge_density": "Arousal",
    "cover_local_contrast": "Arousal",
    "cover_center_border_brightness": "Neither",
    "cover_brightness_centroid_x": "Neither",
    "cover_brightness_centroid_y": "Neither",
}
QUADRANT_ORDER = ("Q1", "Q2", "Q3", "Q4")
QUADRANT_AXES = {
    "Q1": {"Valence": 1, "Arousal": 1},
    "Q2": {"Valence": 0, "Arousal": 1},
    "Q3": {"Valence": 0, "Arousal": 0},
    "Q4": {"Valence": 1, "Arousal": 0},
}
MIN_ABS_SPEARMAN = 0.20
MAX_ADJUSTED_P = 0.05


def _benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return adjusted


def _domain_features(feature_data):
    return {
        "Metadados": [name for name in METADATA_FEATURES if name in feature_data],
        "Áudio": [name for name in feature_data if name.startswith("audio_")],
        "Letras": [name for name in feature_data if name.startswith("lyrics_")],
        "Capas": [name for name in feature_data if name.startswith("cover_")],
    }


def _association_outputs(feature_data, domains):
    unknown_quadrants = sorted(set(feature_data["Quadrant"]) - set(QUADRANT_AXES))
    if unknown_quadrants:
        raise ValueError(f"Unknown quadrant labels: {unknown_quadrants}")

    targets = pd.DataFrame(
        [QUADRANT_AXES[quadrant] for quadrant in feature_data["Quadrant"]],
        index=feature_data.index,
    )
    records = []
    feature_domain = {
        feature: domain for domain, names in domains.items() for feature in names
    }

    for feature, expected_axis in EXPECTED_AXES.items():
        record = {
            "Feature": feature,
            "Domain": feature_domain[feature],
            "Expected axis": expected_axis,
        }
        values = pd.to_numeric(feature_data[feature], errors="coerce")
        for axis in ("Valence", "Arousal"):
            paired = pd.concat([values, targets[axis]], axis=1).dropna()
            if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2:
                rho, p_value = np.nan, 1.0
            else:
                result = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
                rho, p_value = float(result.statistic), float(result.pvalue)
            record[f"rho_{axis.lower()}"] = rho
            record[f"p_{axis.lower()}"] = p_value
        records.append(record)

    associations = pd.DataFrame(records)
    for axis in ("valence", "arousal"):
        associations[f"q_{axis}"] = _benjamini_hochberg(
            associations[f"p_{axis}"].to_numpy()
        )

    def supports_expected_axis(row):
        expected = row["Expected axis"]
        if expected == "Neither":
            return pd.NA
        expected_rho = row[f"rho_{expected.lower()}"]
        other_axis = "arousal" if expected == "Valence" else "valence"
        other_rho = row[f"rho_{other_axis}"]
        expected_q = row[f"q_{expected.lower()}"]
        return bool(
            np.isfinite(expected_rho)
            and abs(expected_rho) >= MIN_ABS_SPEARMAN
            and expected_q < MAX_ADJUSTED_P
            and abs(expected_rho) > abs(other_rho)
        )

    associations["Supports expected axis"] = associations.apply(
        supports_expected_axis, axis=1
    )
    hypotheses = associations.loc[associations["Expected axis"] != "Neither"]
    axis_summary = (
        hypotheses.groupby(["Domain", "Expected axis"], sort=False)
        .agg(
            Features=("Feature", "count"),
            Supported=("Supports expected axis", "sum"),
        )
        .reset_index()
    )
    return associations, axis_summary


def _correlation_outputs(feature_data, domains):
    matrices = {
        domain: feature_data[names].corr(method="spearman")
        for domain, names in domains.items()
        if names
    }
    figure, axes = plt.subplots(2, 2, figsize=(18, 15))
    for axis, (domain, matrix) in zip(axes.flat, matrices.items()):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_title(f"Correlação de Spearman — {domain}")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return matrices, figure


def _separation_outputs(feature_data, domains):
    rows = []
    for domain, names in domains.items():
        for feature in names:
            groups = [
                pd.to_numeric(
                    feature_data.loc[feature_data["Quadrant"] == quadrant, feature],
                    errors="coerce",
                ).dropna().to_numpy()
                for quadrant in QUADRANT_ORDER
            ]
            nonempty = [values for values in groups if len(values)]
            n = sum(len(values) for values in nonempty)
            class_count = len(nonempty)
            if class_count < 2 or n <= class_count:
                statistic, p_value, epsilon_squared = np.nan, 1.0, 0.0
            elif all(np.unique(values).size == 1 for values in nonempty) and len(
                {values[0] for values in nonempty}
            ) == 1:
                statistic, p_value, epsilon_squared = 0.0, 1.0, 0.0
            else:
                test = kruskal(*nonempty)
                statistic, p_value = float(test.statistic), float(test.pvalue)
                epsilon_squared = max(
                    0.0, (statistic - class_count + 1) / (n - class_count)
                )
            rows.append({
                "Domain": domain,
                "Feature": feature,
                "Kruskal H": statistic,
                "p-value": p_value,
                "Epsilon squared": epsilon_squared,
            })

    separation = pd.DataFrame(rows)
    separation["q-value"] = _benjamini_hochberg(separation["p-value"].to_numpy())
    separation = separation.sort_values(
        ["Domain", "Epsilon squared"], ascending=[True, False]
    ).reset_index(drop=True)

    figure, axes = plt.subplots(4, 2, figsize=(15, 16))
    for row_index, (domain, names) in enumerate(domains.items()):
        top_features = (
            separation.loc[separation["Domain"] == domain]
            .nlargest(2, "Epsilon squared")["Feature"]
            .tolist()
        )
        for column_index in range(2):
            axis = axes[row_index, column_index]
            if column_index >= len(top_features):
                axis.set_visible(False)
                continue
            feature = top_features[column_index]
            grouped_values = [
                pd.to_numeric(
                    feature_data.loc[feature_data["Quadrant"] == quadrant, feature],
                    errors="coerce",
                ).dropna().to_numpy()
                for quadrant in QUADRANT_ORDER
            ]
            axis.boxplot(grouped_values, tick_labels=QUADRANT_ORDER, showfliers=False)
            axis.set_title(f"{domain}: {feature}")
            axis.set_xlabel("Quadrante")
            axis.set_ylabel(feature)
    figure.suptitle("Duas características com maior separação por domínio", y=1.01)
    figure.tight_layout()
    return separation, figure


def _index_media(domain, extension):
    index = {}
    for quadrant in (DATA / domain).glob("Q*"):
        for path in quadrant.iterdir():
            if path.is_file() and path.suffix.lower() == extension:
                if path.stem in index:
                    raise ValueError(f"Duplicate media ID in {domain}: {path.stem}")
                index[path.stem] = path
    return index


def _best_key_correlation(chroma_profile, template):
    profile = (chroma_profile - chroma_profile.mean()) / (
        chroma_profile.std() + 1e-12
    )
    template = (template - template.mean()) / (template.std() + 1e-12)
    return max(
        float(np.corrcoef(profile, np.roll(template, shift))[0, 1])
        for shift in range(12)
    )


def _extract_audio(path):
    sample_rate = 22050
    n_fft = 2048
    hop = 512
    signal, sample_rate = librosa.load(path, sr=sample_rate, mono=True)
    if not len(signal):
        raise ValueError(f"Empty audio file: {path}")

    magnitude = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop))
    chroma = librosa.feature.chroma_stft(
        S=magnitude, sr=sample_rate, n_fft=n_fft, hop_length=hop
    )
    chroma_profile = chroma.mean(axis=1)
    chroma_probabilities = chroma_profile / (chroma_profile.sum() + 1e-12)
    chroma_entropy = -float(
        np.sum(chroma_probabilities * np.log2(chroma_probabilities + 1e-12))
    )

    onset_envelope = librosa.onset.onset_strength(
        y=signal, sr=sample_rate, hop_length=hop
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sample_rate, hop_length=hop
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop,
        backtrack=False,
    )
    beat_intervals = np.diff(
        librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop)
    )
    duration_seconds = len(signal) / sample_rate

    rms = librosa.feature.rms(
        S=magnitude, frame_length=n_fft, hop_length=hop
    )[0]
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=sample_rate, roll_percent=0.85
    )[0]
    zero_crossings = librosa.feature.zero_crossing_rate(
        signal, frame_length=n_fft, hop_length=hop
    )[0]

    return {
        "audio_major_minor_profile": (
            _best_key_correlation(chroma_profile, MAJOR_PROFILE)
            - _best_key_correlation(chroma_profile, MINOR_PROFILE)
        ),
        "audio_chroma_entropy_bits": chroma_entropy,
        "audio_tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]),
        "audio_onsets_per_second": len(onset_frames) / duration_seconds,
        "audio_beat_interval_cv": (
            float(np.std(beat_intervals) / np.mean(beat_intervals))
            if len(beat_intervals) > 1 and np.mean(beat_intervals) > 0
            else np.nan
        ),
        "audio_rms_mean": float(np.mean(rms)),
        "audio_spectral_centroid_hz": float(np.mean(centroid)),
        "audio_spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "audio_spectral_rolloff85_hz": float(np.mean(rolloff)),
        "audio_zero_crossing_rate": float(np.mean(zero_crossings)),
    }


def _extract_lyrics(path):
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    lines = [
        re.findall(r"[a-z]+(?:'[a-z]+)?", line.lower())
        for line in text.splitlines()
    ]
    nonempty_lines = [line for line in lines if line]
    if not tokens:
        return {
            "lyrics_positive_rate": np.nan,
            "lyrics_negative_rate": np.nan,
            "lyrics_sentiment_balance": np.nan,
            "lyrics_high_arousal_rate": np.nan,
            "lyrics_low_arousal_rate": np.nan,
            "lyrics_type_token_ratio": np.nan,
            "lyrics_mean_token_length": np.nan,
            "lyrics_punctuation_per_100_words": np.nan,
            "lyrics_nonempty_line_count": 0,
            "lyrics_repeated_line_rate": np.nan,
        }

    word_count = len(tokens)
    positive_rate = 100 * sum(word in POSITIVE_WORDS for word in tokens) / word_count
    negative_rate = 100 * sum(word in NEGATIVE_WORDS for word in tokens) / word_count
    high_rate = 100 * sum(word in HIGH_ACTIVATION_WORDS for word in tokens) / word_count
    low_rate = 100 * sum(word in LOW_ACTIVATION_WORDS for word in tokens) / word_count
    normalized_lines = [" ".join(line) for line in nonempty_lines]
    repeated = Counter(normalized_lines)
    punctuation_rate = 100 * len(re.findall(r"[!?.,;:]", text)) / word_count

    return {
        "lyrics_positive_rate": positive_rate,
        "lyrics_negative_rate": negative_rate,
        "lyrics_sentiment_balance": positive_rate - negative_rate,
        "lyrics_high_arousal_rate": high_rate,
        "lyrics_low_arousal_rate": low_rate,
        "lyrics_type_token_ratio": len(set(tokens)) / word_count,
        "lyrics_mean_token_length": float(np.mean([len(word) for word in tokens])),
        "lyrics_punctuation_per_100_words": punctuation_rate,
        "lyrics_nonempty_line_count": len(nonempty_lines),
        "lyrics_repeated_line_rate": (
            100
            * sum(count for count in repeated.values() if count > 1)
            / len(nonempty_lines)
            if nonempty_lines
            else np.nan
        ),
    }


def _extract_cover(path):
    rgb = np.asarray(
        Image.open(path)
        .convert("RGB")
        .resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hsv = np.asarray(
        Image.fromarray((rgb * 255).astype(np.uint8)).convert("HSV"),
        dtype=np.float64,
    ) / 255.0
    saturation = hsv[..., 1]
    brightness = hsv[..., 2]
    rg = red - green
    yb = (red + green) / 2 - blue
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )

    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    histogram = np.bincount(
        np.rint(gray.ravel() * 255).astype(np.uint8), minlength=256
    ).astype(float)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    dx = np.diff(gray, axis=1, prepend=gray[:, :1])
    dy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.hypot(dx, dy)
    local_difference = (np.abs(dx).sum() + np.abs(dy).sum()) / (
        dx.size + dy.size
    )

    center = brightness[64:192, 64:192]
    border_mask = np.ones(brightness.shape, dtype=bool)
    border_mask[64:192, 64:192] = False
    weights = gray + 1e-12
    x_grid = np.linspace(0, 1, gray.shape[1])[None, :]
    y_grid = np.linspace(0, 1, gray.shape[0])[:, None]

    return {
        "cover_saturation_mean": float(saturation.mean()),
        "cover_brightness_mean": float(brightness.mean()),
        "cover_colorfulness": float(colorfulness),
        "cover_grayscale_std": float(np.std(gray)),
        "cover_grayscale_entropy_bits": entropy,
        "cover_edge_density": float(np.mean(gradient > 0.10)),
        "cover_local_contrast": float(local_difference),
        "cover_center_border_brightness": float(
            center.mean() - brightness[border_mask].mean()
        ),
        "cover_brightness_centroid_x": float((weights * x_grid).sum() / weights.sum()),
        "cover_brightness_centroid_y": float((weights * y_grid).sum() / weights.sum()),
    }


def run_section10():
    """Extract engineered features for the training songs."""
    train = pd.read_csv(TRAIN_FILE)
    if train["Audio_Song"].isna().any() or train["Audio_Song"].duplicated().any():
        raise ValueError("Audio_Song must be present and unique in training metadata.")
    if not set(["Quadrant"] + METADATA_FEATURES).issubset(train.columns):
        raise ValueError("A required metadata feature retained in section 3 is missing.")

    metadata = train[["Audio_Song", "Quadrant"] + METADATA_FEATURES].copy()
    for feature in METADATA_FEATURES:
        metadata[feature] = pd.to_numeric(metadata[feature], errors="coerce")

    audio_index = _index_media("audio", ".mp3")
    lyric_index = _index_media("lyrics", ".txt")
    cover_index = _index_media("album_covers", ".jpg")
    for domain, index, key in [
        ("audio", audio_index, "Audio_Song"),
        ("lyrics", lyric_index, "Lyric_Song"),
    ]:
        missing = sorted(set(train[key].astype(str)) - set(index))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} {domain} files; examples: {missing[:5]}"
            )

    audio_records = []
    for position, row in enumerate(train.itertuples(index=False), start=1):
        audio_id = str(row.Audio_Song)
        path = audio_index[audio_id]
        if path.parent.name != row.Quadrant:
            raise ValueError(f"Audio folder quadrant does not match metadata: {audio_id}")
        audio_records.append({"Audio_Song": audio_id, **_extract_audio(path)})
        if position % 100 == 0:
            print(f"Audio features: {position}/{len(train)}")
    audio = pd.DataFrame(audio_records)

    lyric_records = []
    for row in train.itertuples(index=False):
        audio_id = str(row.Audio_Song)
        lyric_id = str(row.Lyric_Song)
        path = lyric_index[lyric_id]
        if path.parent.name != row.Quadrant:
            raise ValueError(f"Lyrics folder quadrant does not match metadata: {lyric_id}")
        lyric_records.append({"Audio_Song": audio_id, **_extract_lyrics(path)})
    lyrics = pd.DataFrame(lyric_records)

    cover_records = []
    for row in train.itertuples(index=False):
        audio_id = str(row.Audio_Song)
        path = cover_index.get(audio_id)
        if path is None:
            continue
        if path.parent.name != row.Quadrant:
            raise ValueError(f"Cover folder quadrant does not match metadata: {audio_id}")
        cover_records.append({"Audio_Song": audio_id, **_extract_cover(path)})
    cover_features = [
        feature for feature in EXPECTED_AXES if feature.startswith("cover_")
    ]
    covers = pd.DataFrame(cover_records).reindex(
        columns=["Audio_Song"] + cover_features
    )

    feature_data = metadata.merge(audio, on="Audio_Song", validate="one_to_one")
    feature_data = feature_data.merge(lyrics, on="Audio_Song", validate="one_to_one")
    feature_data = feature_data.merge(
        covers, on="Audio_Song", how="left", validate="one_to_one"
    )

    feature_counts = {
        "Audio": sum(feature.startswith("audio_") for feature in EXPECTED_AXES),
        "Cover": sum(feature.startswith("cover_") for feature in EXPECTED_AXES),
        "Lyrics": sum(feature.startswith("lyrics_") for feature in EXPECTED_AXES),
        "Metadata": len(METADATA_FEATURES),
    }
    if any(feature_counts[domain] < 10 for domain in ("Audio", "Lyrics", "Cover")):
        raise ValueError(f"Fewer than 10 features in a required domain: {feature_counts}")
    feature_names = set(feature_data.columns) - {"Audio_Song", "Quadrant"}
    expected_names = set(EXPECTED_AXES)
    if feature_names != expected_names:
        missing = sorted(expected_names - feature_names)
        unexpected = sorted(feature_names - expected_names)
        raise ValueError(
            f"Feature definitions do not match extracted columns; "
            f"missing: {missing}; unexpected: {unexpected}"
        )

    print(
        f"Training songs: {len(train)} | audio: {len(audio)} | lyrics: {len(lyrics)} "
        f"| covers: {len(covers)} | covers missing: {len(train) - len(covers)}"
    )
    print(f"Feature counts: {feature_counts}")
    domains = _domain_features(feature_data)
    associations, axis_summary = _association_outputs(feature_data, domains)
    correlations, correlation_figure = _correlation_outputs(feature_data, domains)
    separation, separation_figure = _separation_outputs(feature_data, domains)
    return {
        "features": feature_data,
        "axis_associations": associations,
        "axis_summary": axis_summary,
        "correlations": correlations,
        "separation": separation,
        "correlation_figure": correlation_figure,
        "separation_figure": separation_figure,
    }
