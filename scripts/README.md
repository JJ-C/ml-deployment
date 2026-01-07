# Scripts Directory

Utility scripts for data processing and setup.

## convert_csv_to_parquet.py

Converts all CSV files in the `data/` directory to Parquet format.

**Usage:**
```bash
python scripts/convert_csv_to_parquet.py
```

**What it does:**
1. Finds all `.csv` files in `data/` directory
2. Reads each CSV file
3. Converts to Parquet with Snappy compression
4. Saves as `.parquet` file alongside the original
5. Reports compression ratios and performance

**Output:**
- Creates `.parquet` files in the same directory
- Prints conversion statistics
- Shows storage savings (typically 50-80%)

**Benefits:**
- 2-7x faster data loading
- 50-80% smaller file size
- Preserves schema and data types
- Production-ready format

**Example output:**
```
Converting: credit_card_fraud_10k.csv
  ✓ Loaded CSV: 10,000 rows, 13 columns (0.05s)
  ✓ Wrote Parquet: credit_card_fraud_10k.parquet (0.02s)
    Size: 0.15 MB (saved 58.3%)

Total space saved: 25.8 MB (65.2%)
✓ Conversion complete!
```

## Future Scripts

Additional utility scripts can be added here:
- Data validation
- Feature engineering
- Dataset generation
- Performance benchmarking
