# Data Engineering Labs

This directory houses scalable data ingestion, transformation, and storage pipelines designed to process hyperscale telemetry and log streams.

## Areas of Focus

- **Streaming Pipelines:** Real-time log ingestion and windowed aggregations using Apache Kafka and Apache Flink.
- **Batch Processing:** High-throughput historical log parsing and ETL using Apache Spark.
- **Data Warehousing & Lakehouses:** Structured data modeling, Iceberg/Delta Lake table maintenance, and dbt transformations.

Processed log streams and cleaned event traces from these pipelines feed directly into the multi-stage anomaly detection models in `../ai-engineering/surprisal-modeling/`.
