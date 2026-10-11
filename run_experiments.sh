set -euo pipefail
python train.py --model linear                                                   --run-name 01_baseline_lineal
python train.py                                                                  --run-name 02_rf_default
python train.py --max-depth 8                                                    --run-name 03_rf_max_depth_8
python train.py --min-samples-leaf 10                                            --run-name 04_rf_min_leaf_10
python train.py --max-features 0.33                                              --run-name 05_rf_max_features_0.33
python train.py --n-estimators 300                                               --run-name 06_rf_300_arboles
python train.py --n-estimators 300 --max-features 0.5 --min-samples-leaf 2       --run-name 07_rf_combinada
python train.py                                                                  --run-name 08_rf_default_repeticion
python train.py --seed 7                                                         --run-name 09_rf_default_seed_7
