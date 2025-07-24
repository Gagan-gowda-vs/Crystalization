# import csv
# from collections import defaultdict
# from multiprocessing import Pool

# # Updated CSV file name
# CSV_FILE = 'synthetic_protein_crystallization_dataset_v2.csv'
# ROW_LIMIT = 13000  # You can adjust this

# def map_features(row):
#     try:
#         method = row['Crystallization_Method'].strip()
#         seq_len = int(row['Sequence_Length']) if row['Sequence_Length'] else 0
#         ph = float(row['pH']) if row['pH'] else None
#         temp = float(row['Temperature_C']) if row['Temperature_C'] else None

#         if not method:
#             return []

#         return [(method, (1, seq_len, ph, temp))]  # (count, total_seq_len, total_ph, total_temp)
#     except Exception:
#         return []

# def reduce_features(mapped_data):
#     summary = defaultdict(lambda: [0, 0, 0.0, 0.0])  # [count, total_seq_len, total_ph, total_temp]

#     for method, (count, seq_len, ph, temp) in mapped_data:
#         summary[method][0] += count
#         summary[method][1] += seq_len
#         if ph is not None:
#             summary[method][2] += ph
#         if temp is not None:
#             summary[method][3] += temp

#     results = []
#     for method, (count, total_seq_len, total_ph, total_temp) in summary.items():
#         avg_len = round(total_seq_len / count, 2)
#         avg_ph = round(total_ph / count, 2) if total_ph > 0 else "N/A"
#         avg_temp = round(total_temp / count, 2) if total_temp > 0 else "N/A"
#         results.append((method, count, avg_len, avg_ph, avg_temp))

#     return sorted(results, key=lambda x: x[1], reverse=True)

# def run_crystoper_mapreduce():
#     with open(CSV_FILE, mode='r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         rows = [row for i, row in enumerate(reader) if i < ROW_LIMIT]

#     with Pool() as pool:
#         mapped = pool.map(map_features, rows)

#     flat_mapped = [item for sublist in mapped for item in sublist]
#     reduced = reduce_features(flat_mapped)

#     print("\nTop Crystallization Methods Summary:\n")
#     for i, (method, count, avg_len, avg_ph, avg_temp) in enumerate(reduced[:10], 1):
#         print(f"{i}. Method: {method}")
#         print(f"   Trials: {count}")
#         print(f"   Avg Sequence Length: {avg_len}")
#         print(f"   Avg pH: {avg_ph}")
#         print(f"   Avg Temp: {avg_temp}°C")

# if __name__ == "__main__":
#     run_crystoper_mapreduce()




import csv
import time
from collections import defaultdict
from multiprocessing import Pool

# Updated CSV file name
CSV_FILE = 'synthetic_protein_crystallization_dataset_v2.csv'
ROW_LIMIT = 13000  # You can adjust this

def map_features(row):
    try:
        method = row['Crystallization_Method'].strip()
        seq_len = int(row['Sequence_Length']) if row['Sequence_Length'] else 0
        ph = float(row['pH']) if row['pH'] else None
        temp = float(row['Temperature_C']) if row['Temperature_C'] else None

        if not method:
            return []

        return [(method, (1, seq_len, ph, temp))]  # (count, total_seq_len, total_ph, total_temp)
    except Exception:
        return []

def reduce_features(mapped_data):
    summary = defaultdict(lambda: [0, 0, 0.0, 0.0])  # [count, total_seq_len, total_ph, total_temp]

    for method, (count, seq_len, ph, temp) in mapped_data:
        summary[method][0] += count
        summary[method][1] += seq_len
        if ph is not None:
            summary[method][2] += ph
        if temp is not None:
            summary[method][3] += temp

    results = []
    for method, (count, total_seq_len, total_ph, total_temp) in summary.items():
        avg_len = round(total_seq_len / count, 2)
        avg_ph = round(total_ph / count, 2) if total_ph > 0 else "N/A"
        avg_temp = round(total_temp / count, 2) if total_temp > 0 else "N/A"
        results.append((method, count, avg_len, avg_ph, avg_temp))

    return sorted(results, key=lambda x: x[1], reverse=True)

def run_crystoper_mapreduce():
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for i, row in enumerate(reader) if i < ROW_LIMIT]

    with Pool() as pool:
        mapped = pool.map(map_features, rows)

    flat_mapped = [item for sublist in mapped for item in sublist]
    reduced = reduce_features(flat_mapped)

    print("\nTop Crystallization Methods Summary:\n")
    for i, (method, count, avg_len, avg_ph, avg_temp) in enumerate(reduced[:10], 1):
        print(f"{i}. Method: {method}")
        print(f"   Trials: {count}")
        print(f"   Avg Sequence Length: {avg_len}")
        print(f"   Avg pH: {avg_ph}")
        print(f"   Avg Temp: {avg_temp}°C")

if __name__ == "__main__":
    start_time = time.time()
    run_crystoper_mapreduce()
    end_time = time.time()
    print(f"\nExecution Time: {round(end_time - start_time, 4)} seconds")
