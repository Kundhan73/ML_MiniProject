#!/usr/bin/env bash
# Fetch the data into data/raw/.
#
# Full results (recommended): Kaggle's Home Credit Default Risk files.
#   1. Accept the rules at https://www.kaggle.com/competitions/home-credit-default-risk/rules
#   2. Download and unzip the data into data/raw/kaggle/  (application_train.csv, bureau.csv,
#      bureau_balance.csv, previous_application.csv, installments_payments.csv,
#      POS_CASH_balance.csv, credit_card_balance.csv)
#      With the Kaggle CLI:  kaggle competitions download -c home-credit-default-risk -p data/raw/kaggle
#                            unzip data/raw/kaggle/home-credit-default-risk.zip -d data/raw/kaggle
#   3. python -m src.history
#
# Without a Kaggle account, this script fetches a public GitHub mirror of the application
# table only (215,257 applicants). Everything runs, but without history features
# (expect ROC-AUC ~0.75 instead of ~0.79).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ -f data/raw/kaggle/application_train.csv ]; then
  echo "Kaggle data found in data/raw/kaggle/. Run: python -m src.history"; exit 0
fi
if [ -f data/raw/home_credit_train.csv ]; then
  echo "Mirror already present in data/raw/"; exit 0
fi
REPO="https://github.com/sultanbeishenkulov/home-credit-default-risk"
COMMIT="654f528045a4087b446ee6623b7fdf99f3d9b803"
TMP="$(mktemp -d)"
echo "No Kaggle data found. Fetching the application-table mirror from $REPO"
git clone -q --filter=blob:none --no-checkout "$REPO" "$TMP/mirror"
git -C "$TMP/mirror" checkout -q "$COMMIT" -- train.csv.zip
unzip -o -q "$TMP/mirror/train.csv.zip" train.csv -d "$TMP"
mv "$TMP/train.csv" data/raw/home_credit_train.csv
rm -rf "$TMP"
echo "Saved data/raw/home_credit_train.csv"
