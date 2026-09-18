import pandas as pd
import numpy as np
import os
import time
import shutil
import hashlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv(r"C:\Users\user\OneDrive\Documents\SSM\training_data.csv")

action = {"KEEP": 0, "REVIEW": 1, "CLEANUP": 2}
df["action_label"] = df["action_label"].map(action)

x = df[["file_size_mb", "days_since_modified", "days_since_accessed", "duplicate_count"]]
y = df["action_label"]

classification_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
classification_model.fit(x.values, y)

def calculate_duplicate_counts(target_folder_path):
    file_hashes = {}
    duplicate_counts = {}
    for root, folders, files in os.walk(target_folder_path):
        for file_name in files:
            full_file_path = os.path.join(root, file_name)
            try:
                hash_object = hashlib.md5()
                with open(full_file_path, "rb") as file:
                    while True:
                        chunk = file.read(4096)
                        if not chunk:
                            break
                        hash_object.update(chunk)
                file_hash = hash_object.hexdigest()
                if file_hash not in file_hashes:
                    file_hashes[file_hash] = []
                file_hashes[file_hash].append(full_file_path)
            except:
                continue

    for file_hash, file_list in file_hashes.items():
        count = len(file_list)
        for file_path in file_list:
            duplicate_counts[file_path] = count - 1
    return duplicate_counts

def scan_and_classify_live_pc(target_folder_path):
    print("\n" + "=" * 60)
    print("STORAGE SCAN STARTED")
    print("Folder:", target_folder_path)
    if not os.path.exists(target_folder_path):
        print("Folder does not exist!")
        return
    
    current_time = time.time()
    delete_list = []
    review_list = []
    keep_count = 0

    duplicate_counts = calculate_duplicate_counts(target_folder_path)

    for root, folders, files in os.walk(target_folder_path):
        for file_name in files:
            full_file_path = os.path.join(root, file_name)
            try:
                file_properties = os.stat(full_file_path)
                size_mb = file_properties.st_size / (1024 * 1024)
                
                # DAYS SINCE MODIFIED
                days_old = (current_time - file_properties.st_mtime) / (24 * 3600)
                # DAYS SINCE ACCESSED
                days_unaccessed = (current_time - file_properties.st_atime) / (24 * 3600)              
                # DUPLICATE COUNT
                duplicates = duplicate_counts.get(full_file_path, 0)

                input_features = pd.DataFrame([[size_mb, days_old, days_unaccessed, duplicates]],
                    columns=["file_size_mb", "days_since_modified", "days_since_accessed", "duplicate_count"])
                prediction = classification_model.predict(input_features.values)[0]

                reason_text = "Flagged by Gradient Boosting Classifier based on historical feature pattern correlations."
                if duplicates > 0:
                    reason_text = f"This file has {duplicates} exact duplicate copy/copies elsewhere on this path."
                elif size_mb > 100 and days_unaccessed > 90:
                    reason_text = "This is a heavy space-consuming file and hasn't been accessed in over 3 months."
                elif days_unaccessed > 180:
                    reason_text = "This file hasn't been opened or accessed in over 6 months."
                elif size_mb > 150:
                    reason_text = "This file volume size is exceptionally large (Greater than 150 MB)."

                file_info = {
                    "name": file_name, 
                    "path": full_file_path, 
                    "size": round(size_mb, 2),
                    "reason": reason_text 
                }

                if prediction == 2:
                    delete_list.append(file_info)
                elif prediction == 1:
                    review_list.append(file_info)
                else:
                    keep_count += 1
            except:
                continue

    print("\n" + "=" * 60)
    print("SCAN RESULT")
    print("=" * 60)
    print("Files to keep   :", keep_count)
    print("Files to review :", len(review_list))
    print("Files to delete :", len(delete_list))

    print("\nFILES TO DELETE")
    if delete_list:
        for number, item in enumerate(delete_list, 1):
            print(number, item["name"], "(", item["size"], "MB)")
            print("Path:", item["path"])
            print("Reason:", item["reason"]) 
    else:
        print("No files to delete")

    print("\nFILES TO REVIEW")
    if review_list:
        for number, item in enumerate(review_list, 1):
            print(number, item["name"], "(", item["size"], "MB)")
            print("Path:", item["path"])
            print("Reason:", item["reason"])
    else:
        print("No files to review")

    # VISUALIZATION
    cleanup_count = len(delete_list)
    review_count = len(review_list)

    labels = ["KEEP", "REVIEW", "CLEANUP"]
    counts = [keep_count, review_count, cleanup_count]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts)
    plt.title("File Classification Results")
    plt.xlabel("Action")
    plt.ylabel("Number of Files")

    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha="center")
    plt.tight_layout()
    plt.show()
    print("=" * 60)

scan_and_classify_live_pc(r"C:\Users\user\Desktop\SSM_Test")
