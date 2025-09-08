#!/usr/bin/env python3
"""
Identify two subclusters within the malignant cohort and train a simple
linear classifier in original feature space to separate them. Report
standardized feature importances (logistic regression coefficients on
z-scored features).

Usage:
  python examples/breast_cancer/train_malignant_subclusters.py \
         --csv examples/breast_cancer/breast_cancer_wdbc.csv \
         --penalty l2 --cv 5 --balanced

Outputs:
  - *_malignant_subcluster_linear_importance.csv: ranked feature importances
  - (optional) *_malignant_subcluster_assignments.csv: malignant rows with cluster labels
"""

import os
import sys
import argparse


def main():
    try:
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import silhouette_score
    except Exception as e:
        print("Error: requires scikit-learn, pandas, and numpy.\n"
              "Install with: pip install scikit-learn pandas numpy", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Malignant subclusters + linear classifier feature importances")
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'breast_cancer_wdbc.csv'),
                        help='Path to CSV with numeric label column and features')
    parser.add_argument('--label-col', default='label', help='Name of the label column (default: label)')
    parser.add_argument('--malignant-label', type=int, default=1, help='Numeric code for malignant class (default: 1)')
    parser.add_argument('--penalty', choices=['l1', 'l2'], default='l2', help='Regularization penalty (default: l2)')
    parser.add_argument('--cv', type=int, default=5, help='CV folds (default: 5)')
    parser.add_argument('--balanced', action='store_true', help='Use class_weight=balanced')
    parser.add_argument('--max-iter', type=int, default=5000, help='Max iterations (default: 5000)')
    parser.add_argument('--assignments-output', default=None,
                        help='Optional CSV path to write malignant rows with subcluster labels')
    parser.add_argument('--output', default=None,
                        help='Output CSV path for ranked feature importances')
    parser.add_argument('--top', type=int, default=20, help='Print top-N features (default: 20)')
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        print(f"Error: label column '{args.label_col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)
    feature_cols = [c for c in df.columns if c != args.label_col]
    if not feature_cols:
        print("Error: no feature columns found", file=sys.stderr)
        sys.exit(2)

    # Filter malignant cohort
    mal_df = df[df[args.label_col].astype(int) == int(args.malignant_label)].copy()
    if len(mal_df) < 4:
        print("Error: too few malignant samples to find subclusters", file=sys.stderr)
        sys.exit(2)
    X_mal = mal_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    # Identify two subclusters (KMeans on standardized features)
    scaler_clust = StandardScaler(with_mean=True, with_std=True)
    Xc = scaler_clust.fit_transform(X_mal)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    sublabels = kmeans.fit_predict(Xc)  # 0/1 subclusters within malignant

    # Silhouette score for diagnostic
    try:
        sil = silhouette_score(Xc, sublabels, metric='euclidean')
    except Exception:
        sil = None

    # Optional: write assignments
    if args.assignments_output is None:
        out_dir = os.path.dirname(os.path.abspath(args.csv))
        base = os.path.splitext(os.path.basename(args.csv))[0]
        assign_path = os.path.join(out_dir, base + '_malignant_subcluster_assignments.csv')
    else:
        assign_path = args.assignments_output
    try:
        mal_out = mal_df.copy()
        mal_out['subcluster'] = sublabels
        mal_out.to_csv(assign_path, index=False)
        wrote_assign = True
    except Exception:
        wrote_assign = False

    # Train linear classifier to separate the two malignant subclusters
    y_sub = sublabels.astype(int)

    class_weight = 'balanced' if args.balanced else None
    solver = 'saga'
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    logitcv = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        penalty=args.penalty,
        solver=solver,
        max_iter=args.max_iter,
        class_weight=class_weight,
        scoring='roc_auc',
        n_jobs=None,
        refit=True,
    )
    # Pipeline: Standardize then logistic CV (coeffs are on standardized features)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('logit', logitcv)
    ])
    pipe.fit(X_mal, y_sub)

    # Extract coefficients from final classifier
    clf = logitcv
    coefs = clf.coef_.ravel()
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)

    ranked = []
    for rank, idx in enumerate(order, 1):
        ranked.append({
            'rank': rank,
            'feature': feature_cols[idx],
            'coef': float(coefs[idx]),
            'abs_coef': float(abs_coefs[idx])
        })

    # Determine output path
    if args.output is None:
        out_dir = os.path.dirname(os.path.abspath(args.csv))
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join(out_dir, base + '_malignant_subcluster_linear_importance.csv')
    else:
        out_path = args.output

    # Write CSV
    try:
        pd.DataFrame(ranked).to_csv(out_path, index=False)
        wrote_importances = True
    except Exception:
        wrote_importances = False

    # Summary
    n0 = int(np.sum(y_sub == 0))
    n1 = int(np.sum(y_sub == 1))
    print(f"\nMalignant cohort size: {len(X_mal)} | subcluster sizes: {n0} vs {n1}")
    if sil is not None:
        print(f"Silhouette score (k=2): {sil:.3f}")
    try:
        best_C = float(clf.C_[0]) if hasattr(clf, 'C_') else None
        print(f"Selected C: {best_C}")
    except Exception:
        pass
    print(f"Penalty: {args.penalty} | Balanced: {bool(args.balanced)}")
    print(f"Top {min(args.top, len(ranked))} features by |standardized coef|:")
    for row in ranked[:args.top]:
        sgn = '+' if row['coef'] >= 0 else '-'
        print(f"  {row['rank']:>3d}. {row['feature']:<24s} {sgn}{abs(row['coef']):.4f}")
    if wrote_importances:
        print(f"\n✓ Wrote ranked importances to: {out_path}")
    if wrote_assign:
        print(f"✓ Wrote subcluster assignments to: {assign_path}")


if __name__ == '__main__':
    main()

