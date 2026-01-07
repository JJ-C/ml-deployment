#!/usr/bin/env python3

import pandas as pd
import os
from pathlib import Path
import time

print("=" * 60)
print("CSV to Parquet Conversion Script")
print("=" * 60)

data_dir = Path(__file__).parent.parent / "data"

csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    print("\n✗ No CSV files found in data directory")
    exit(1)

print(f"\nFound {len(csv_files)} CSV file(s) to convert:")
for csv_file in csv_files:
    print(f"  - {csv_file.name} ({csv_file.stat().st_size / 1024 / 1024:.2f} MB)")

print("\n" + "=" * 60)
print("Starting conversion...")
print("=" * 60)

results = []

for csv_file in csv_files:
    parquet_file = csv_file.with_suffix('.parquet')
    
    print(f"\n[{csv_files.index(csv_file) + 1}/{len(csv_files)}] Converting: {csv_file.name}")
    
    try:
        start_time = time.time()
        
        df = pd.read_csv(csv_file)
        load_time = time.time() - start_time
        
        rows, cols = df.shape
        csv_size = csv_file.stat().st_size
        
        print(f"  ✓ Loaded CSV: {rows:,} rows, {cols} columns ({load_time:.2f}s)")
        
        write_start = time.time()
        df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False)
        write_time = time.time() - write_start
        
        parquet_size = parquet_file.stat().st_size
        compression_ratio = (1 - parquet_size / csv_size) * 100
        
        print(f"  ✓ Wrote Parquet: {parquet_file.name} ({write_time:.2f}s)")
        print(f"    Size: {parquet_size / 1024 / 1024:.2f} MB (saved {compression_ratio:.1f}%)")
        
        results.append({
            "file": csv_file.name,
            "rows": rows,
            "csv_size_mb": csv_size / 1024 / 1024,
            "parquet_size_mb": parquet_size / 1024 / 1024,
            "compression": compression_ratio,
            "load_time": load_time,
            "write_time": write_time
        })
        
    except Exception as e:
        print(f"  ✗ Error converting {csv_file.name}: {e}")
        results.append({
            "file": csv_file.name,
            "error": str(e)
        })

print("\n" + "=" * 60)
print("Conversion Summary")
print("=" * 60)

successful = [r for r in results if "error" not in r]
failed = [r for r in results if "error" in r]

if successful:
    print(f"\n✓ Successfully converted {len(successful)} file(s):\n")
    
    total_csv_size = sum(r["csv_size_mb"] for r in successful)
    total_parquet_size = sum(r["parquet_size_mb"] for r in successful)
    total_saved = ((total_csv_size - total_parquet_size) / total_csv_size) * 100
    
    for result in successful:
        print(f"  {result['file']}")
        print(f"    Rows: {result['rows']:,}")
        print(f"    CSV: {result['csv_size_mb']:.2f} MB → Parquet: {result['parquet_size_mb']:.2f} MB")
        print(f"    Compression: {result['compression']:.1f}%")
        print(f"    Read time: {result['load_time']:.2f}s, Write time: {result['write_time']:.2f}s")
        print()
    
    print(f"Total space saved: {total_csv_size - total_parquet_size:.2f} MB ({total_saved:.1f}%)")

if failed:
    print(f"\n✗ Failed to convert {len(failed)} file(s):")
    for result in failed:
        print(f"  {result['file']}: {result['error']}")

print("\n" + "=" * 60)
print("✓ Conversion complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Update scripts to use .parquet files")
print("2. Test model training with Parquet data")
print("3. Optionally remove CSV files if no longer needed")
print("=" * 60)
