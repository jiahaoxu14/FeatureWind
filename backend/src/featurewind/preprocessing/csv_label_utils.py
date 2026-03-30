from __future__ import annotations

import csv


def read_csv_frame(path: str):
    import pandas as pd

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return pd.DataFrame()

    has_header = False
    for cell in first:
        try:
            float(cell)
        except Exception:
            has_header = True
            break

    if has_header:
        return pd.read_csv(path)

    headers = [f"Feature {i}" for i in range(len(first))]
    return pd.read_csv(path, header=None, names=headers)


def identify_label_column(df) -> str | None:
    label_candidates = ["label", "class", "target", "y"]
    for candidate in label_candidates:
        for actual_col in df.columns:
            if str(actual_col).lower() == candidate:
                return actual_col

    if len(df.columns) == 0:
        return None

    last_col = df.columns[-1]
    try:
        import pandas as pd

        pd.to_numeric(df[last_col])
    except Exception:
        return last_col
    return None


def load_csv_features_and_labels(path: str):
    import pandas as pd

    df = read_csv_frame(path)
    label_column = identify_label_column(df)

    if label_column is not None:
        point_labels = df[label_column].tolist()
        feature_df = df.drop(columns=[label_column])
    else:
        point_labels = None
        feature_df = df

    feature_df = feature_df.apply(pd.to_numeric, errors="raise")
    return feature_df.values.tolist(), [str(col) for col in feature_df.columns], point_labels


def humanize_point_labels(col_labels, point_labels):
    if not isinstance(col_labels, list) or not isinstance(point_labels, list) or not point_labels:
        return point_labels

    dataset_label_maps = {
        (
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280_od315",
            "proline",
        ): {
            "1": "Barolo",
            "2": "Grignolino",
            "3": "Barbera",
        },
        (
            "area",
            "perimeter",
            "compactness",
            "kernel_length",
            "kernel_width",
            "asymmetry_coefficient",
            "groove_length",
        ): {
            "1": "Kama",
            "2": "Rosa",
            "3": "Canadian",
        },
    }

    label_map = dataset_label_maps.get(tuple(col_labels))
    if label_map is not None:
        return [label_map.get(str(label), str(label)) for label in point_labels]

    return point_labels
