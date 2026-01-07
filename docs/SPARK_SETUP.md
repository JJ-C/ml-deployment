# Spark Configuration for Feast Materialization

This guide explains how to use Spark as the compute engine for Feast feature materialization.

## Overview

Spark provides distributed processing for large-scale feature materialization, offering:
- **Scalability**: Handle TB/PB-scale datasets
- **Performance**: Parallel processing across cluster nodes
- **Efficiency**: Optimized Parquet reading and Cassandra writes

## Local Development Setup

### 1. Install Dependencies

```bash
pip install -r requirements-spark.txt
```

### 2. Run Materialization with Spark

```bash
python scripts/feast_materialize_spark.py
```

The script will:
- Initialize a local Spark session
- Load feature definitions from Feast
- Use Spark to read Parquet files
- Materialize features to Cassandra online store
- Provide Spark UI at http://localhost:4040

## Production Deployment

### Option 1: Databricks

```yaml
batch_engine:
  type: spark
  spark_conf:
    spark.master: "databricks"
    spark.databricks.cluster.id: "your-cluster-id"
```

### Option 2: AWS EMR

```yaml
batch_engine:
  type: spark
  spark_conf:
    spark.master: "yarn"
    spark.executor.instances: "10"
    spark.executor.memory: "8g"
```

### Option 3: Self-Managed Cluster

See `feature_store.spark_cluster.yaml` for full production configuration.

```bash
# Copy production config
cp feature_repo/feature_store.spark_cluster.yaml feature_repo/feature_store.yaml

# Update with your cluster details
# - Spark master URL
# - Cassandra hosts
# - Resource allocation
```

## Configuration Parameters

### Resource Allocation

| Parameter | Local Dev | Production | Description |
|-----------|-----------|------------|-------------|
| `spark.executor.memory` | 4g | 8-16g | Memory per executor |
| `spark.driver.memory` | 2g | 4-8g | Driver memory |
| `spark.executor.instances` | N/A | 10-50 | Number of executors |
| `spark.executor.cores` | N/A | 4-8 | Cores per executor |

### Performance Tuning

```yaml
# Optimize for large datasets
spark.sql.shuffle.partitions: "200"
spark.sql.adaptive.enabled: "true"

# Enable Arrow for faster Pandas conversion
spark.sql.execution.arrow.pyspark.enabled: "true"

# Parquet optimization
spark.sql.parquet.filterPushdown: "true"
spark.sql.parquet.compression.codec: "snappy"
```

## Monitoring

### Spark UI
- Local: http://localhost:4040
- Cluster: http://<driver-host>:4040

### Key Metrics to Monitor
- **Stage duration**: Time for each materialization stage
- **Shuffle read/write**: Data movement between executors
- **Task failures**: Retry and failure counts
- **Memory usage**: Executor and driver memory

## Troubleshooting

### Out of Memory Errors
```yaml
# Increase executor memory
spark.executor.memory: "16g"
spark.driver.memory: "8g"

# Increase partitions to reduce per-task data
spark.sql.shuffle.partitions: "400"
```

### Slow Performance
```yaml
# Enable adaptive query execution
spark.sql.adaptive.enabled: "true"

# Increase parallelism
spark.default.parallelism: "200"
spark.executor.instances: "20"
```

### Connection Timeouts
```yaml
# Increase network timeout
spark.network.timeout: "800s"
spark.executor.heartbeatInterval: "60s"
```

## Cost Optimization

### Use Spot Instances (AWS)
- Configure EMR with spot instances for executors
- Keep driver on on-demand instance

### Dynamic Allocation
```yaml
spark.dynamicAllocation.enabled: "true"
spark.dynamicAllocation.minExecutors: "2"
spark.dynamicAllocation.maxExecutors: "20"
```

### Schedule During Off-Peak Hours
- Run materialization during low-traffic periods
- Use cron jobs or Airflow for scheduling

## Comparison: Local vs Spark

| Aspect | Local (Pandas) | Spark |
|--------|---------------|-------|
| Dataset size | < 1GB | > 100GB |
| Processing time (10GB) | ~30 min | ~5 min |
| Memory required | 3x dataset size | Distributed |
| Setup complexity | Simple | Moderate |
| Cost | Free | Cluster costs |
| Best for | Dev/testing | Production |

## Next Steps

1. Test locally: `python scripts/feast_materialize_spark.py`
2. Monitor Spark UI to understand performance
3. Configure production cluster settings
4. Set up scheduled materialization (Airflow/cron)
5. Monitor and optimize based on metrics
