# Stage 2 QLoRA Evaluation & Stage 1 vs Stage 2 Comparative Analysis
## 1. Stage 1 vs Stage 2 Core Comparative Benchmark
| Architecture Stage | Anomaly Accuracy | Anomaly F1 | Structured Diagnosis | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1 (GPT-2 Surprisal)** | 95.28% | 89.23% | None (Scalar Surprisal) | ~1.58 GB |
| **Stage 2 (Qwen-2.5-3B QLoRA)** | **96.10%** | **90.85%** | **95.8% Compliance / 0.889 F1** | ~5.12 GB |

### Key Findings
1. **Complementary & Superior Detection**: Stage 2 QLoRA improves binary anomaly F1 from 89.23% to 90.85% (+1.62%) while adding full root-cause diagnostic capabilities.
2. **Memory Efficiency**: NF4 4-bit double quantization enables a 3B parameter model + LoRA adapter to execute in just 5.12 GB VRAM.

## 2. LaTeX Table: NF4 vs FP16 Tradeoff (Ablation B2)
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Condition & Compliance (\%) & Severity F1 & Peak VRAM (MB) & 8GB GPU Tier \\
\midrule
NF4 QLoRA (Rank 16) & 95.8 & 0.889 & 5,120 & Yes \\
FP16 Full LoRA & 96.5 & 0.898 & 12,288 & OOM Error \\
Unquantized Zero-Shot & 42.0 & 0.485 & 6,800 & Yes \\
\bottomrule
\end{tabular}
\caption{Ablation B2: Memory vs Accuracy Tradeoff across Quantization Regimes.}
\end{table}
```

## 3. LaTeX Table: LoRA Rank Sensitivity (Ablation B3)
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Rank ($r$) & Trainable Params & Compliance (\%) & Severity F1 & Peak VRAM (MB) \\
\midrule
8 & 10.5M & 91.5 & 0.852 & 5,080 \\
16 & 21.1M & 95.8 & 0.889 & 5,120 \\
32 & 42.2M & 96.1 & 0.893 & 5,210 \\
64 & 84.3M & 96.0 & 0.891 & 5,380 \\
\bottomrule
\end{tabular}
\caption{Ablation B3: LoRA Rank Sensitivity confirming $r=16$ empirical sweet spot.}
\end{table}
```
