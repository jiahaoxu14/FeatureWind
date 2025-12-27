#!/usr/bin/env python3
"""
Train a simple linear classifier on the breast cancer dataset and report
standardized feature importances (coefficients on standardized features).

Usage:
  python examples/breast_cancer/train_linear_classifier.py \
         --csv examples/breast_cancer/breast_cancer_wdbc.csv \
         --penalty l2 --cv 5 --balanced

Outputs:
  - Writes a CSV with ranked feature importances next to the data file
    (default: *_linear_feature_importance.csv)
  - Prints the top features to stdout
"""

import os
import sys
import argparse


def main():
    try:
        import pandas as pd
        import numpy as np
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:
        print("Error: requires scikit-learn, pandas, and numpy.\n"
              "Install with: pip install scikit-learn pandas numpy", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Linear classifier + standardized feature importances")
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'breast_cancer_wdbc.csv'),
                        help='Path to CSV with a numeric label column and feature columns')
    parser.add_argument('--label-col', default='label', help='Name of the label column (default: label)')
    parser.add_argument('--penalty', choices=['l1', 'l2'], default='l2', help='Regularization penalty (default: l2)')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--balanced', action='store_true', help='Use class_weight=balanced')
    parser.add_argument('--max-iter', type=int, default=5000, help='Max iterations for solver (default: 5000)')
    parser.add_argument('--output', default=None, help='Output CSV path for ranked importances')
    parser.add_argument('--top', type=int, default=20, help='Print top-N features (default: 20)')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        print(f"Error: label column '{args.label_col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    y = df[args.label_col].astype(int).values
    feature_cols = [c for c in df.columns if c != args.label_col]
    if len(feature_cols) == 0:
        print("Error: no feature columns found", file=sys.stderr)
        sys.exit(2)
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    # Build pipeline: Standardize then LogisticRegressionCV
    # Saga supports both l1 and l2 with binary labels
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

    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), logitcv)
    pipe.fit(X, y)

    # Extract standardized coefficients (on scaled features)
    clf = logitcv  # last step
    coefs = clf.coef_.ravel()  # shape (n_features,)

    # Package results
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

    out_dir = os.path.dirname(os.path.abspath(args.csv))
    out_path = args.output or os.path.join(out_dir, os.path.splitext(os.path.basename(args.csv))[0] + '_linear_feature_importance.csv')

    try:
        pd.DataFrame(ranked).to_csv(out_path, index=False)
    except Exception as e:
        print(f"Warning: failed to write CSV to {out_path}: {e}", file=sys.stderr)
        out_path = None

    # Print summary
    print(f"\nTraining complete on {len(y)} samples, {len(feature_cols)} features")
    try:
        best_C = float(clf.C_[0]) if hasattr(clf, 'C_') else None
        print(f"Selected C: {best_C}")
    except Exception:
        pass
    print(f"Penalty: {args.penalty} | Balanced: {bool(args.balanced)}")
    print(f"Top {args.top} features by |standardized coef|:")
    for row in ranked[:args.top]:
        sgn = '+' if row['coef'] >= 0 else '-'
        print(f"  {row['rank']:>3d}. {row['feature']:<24s} {sgn}{abs(row['coef']):.4f}")

    if out_path:
        print(f"\n✓ Wrote ranked importances to: {out_path}")


if __name__ == '__main__':
    main()

