# Systems Engineering Labs

An enterprise-grade repository housing production architectures across Data Engineering, AI Engineering, and Distributed Systems.

---

## Repository Structure

```text
systems-engineering-labs/
│
├── data-engineering/                    # Scalable data ingestion, ETL, and streaming pipelines
│   ├── streaming/                       # Real-time processing frameworks (Kafka, Flink)
│   ├── batch/                           # High-throughput historical batch ETL (Spark)
│   └── README.md
│
├── ai-engineering/                      # Advanced machine learning and statistical modeling
│   ├── surprisal-modeling/              # Unsupervised log anomaly detection via statistical surprisal
│   │   ├── stage1-gpt2/                 # Stage 1: Unsupervised baseline with GPT-2 Small
│   │   ├── stage2-qlora/                # Stage 2: 4-bit QLoRA diagnostic layer with Qwen-2.5-3B
│   │   └── stage3-mamba/                # Stage 3: Hybrid MambaLog and Mamba S6 streaming architecture
│   └── README.md
│
└── shared/                              # Cross-cutting utilities, schemas, and infrastructure
    ├── schemas/                         # Standardized data contracts (Protobuf, Avro, JSON Schema)
    ├── devops/                          # Containerization, CI/CD workflows, and Terraform configs
    └── README.md
```

---

## Architectural Principles

1. **Modular Decoupling:** Clear separation between upstream data transformation pipelines (`data-engineering/`) and downstream machine learning execution engines (`ai-engineering/`).
2. **Unified Data Contracts:** Telemetry streams adhere to strict schema definitions stored in `shared/schemas/`, ensuring reliable interoperability and deterministic ingestion.
3. **Hyperscale Performance:** Designed for distributed execution environments with sub-linear memory scaling and predictable resource utilization.
