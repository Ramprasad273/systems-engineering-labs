# Ablation and analysis scripts for the surprisal-gpt2 paper submission.
#
# Scripts (run in this order after train.py + evaluate.py):
#   1. scripts/token_stability_check.py   — verify no [UNK] in masking pipeline
#   2. scripts/threshold_sensitivity.py   — precision-recall vs. k analysis
#   3. scripts/ablation_vocab.py          — vocab size V ∈ {500,1K,2K,5K,10K}
#   4. scripts/ablation_depth.py          — model depth L ∈ {2,4,8,12}
#   5. scripts/analyze_results.py         — render all paper tables
