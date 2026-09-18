import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# SYSTEM DEPENDENCY (Replace with your actual model imports)
try:
    from SSM_classification import classification_model, calculate_duplicate_counts
except ImportError:
    classification_model = None
    def calculate_duplicate_counts(folder):
        return {}
try:
    from SSM_regression import history_file
except ImportError:
    history_file = None

# Configure page
st.set_page_config(
    page_title="Smart Storage Manager",
    layout="wide"
)

# Custom CSS for blue theme
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #ffffff;
        }
        h1 {
            text-align: center;
            color: #2b6cb0;
            font-weight: 700;
            margin-top: 20px;
        }
        h3 {
            color: #2b6cb0;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        h5 {
            color: #3182ce;
        }
        hr {
            border: 1px solid #cbd5e0;
            margin-top: 30px;
            margin-bottom: 30px;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #ebf8ff;
            padding: 20px;
            border-radius: 12px;
        }
        /* Metric boxes styling */
        .metric {
            background-color: #ebf8ff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(43, 108, 176, 0.1);
        }
        /* Dataframes styling */
        .dataframe {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(43, 108, 176, 0.1);
        }
        /* Buttons styling */
        button {
            background-color: #3182ce;
            color: #fff;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
        }
        button:hover {
            background-color: #2b6cb0;
        }
        /* Metric text color */
        .stMetric {
            font-weight: bold;
            color: #2b6cb0;
        }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<h1>SMART STORAGE MANAGER (SSM)</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #4a5568;'>Infrastructure Diagnostic Pipeline & Predictive Capacity Forecasting</h5>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar for directory input
st.sidebar.header("System Directory Control")
target_folder = st.sidebar.text_input("Target Directory Path:", value=r"C:\Users\user\Desktop\SSM_Test")
run_scan = st.sidebar.button("Run System Scan")

# Storage Info
total, used, free = shutil.disk_usage("C:")
total_gb = round(total / (1024 ** 3), 2)
used_gb = round(used / (1024 ** 3), 2)
free_gb = round(free / (1024 ** 3), 2)
current_month = datetime.now().strftime("%Y-%m")

# Run scan when button clicked
if run_scan:
    if not os.path.exists(target_folder):
        st.sidebar.error("Selected directory does not exist!")
    else:
        with st.spinner("Processing files & running ML models..."):
            current_time = time.time()
            delete_list = []
            review_list = []
            keep_count = 0

            # Calculate duplicate counts
            duplicate_counts = calculate_duplicate_counts(target_folder)

            # Walk through files
            for root, folders, files in os.walk(target_folder):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    try:
                        props = os.stat(full_path)
                        size_mb = props.st_size / (1024 * 1024)
                        days_old = (current_time - props.st_mtime) / (24 * 3600)
                        days_unaccessed = (current_time - props.st_atime) / (24 * 3600)
                        duplicates = duplicate_counts.get(full_path, 0)

                        features = pd.DataFrame([[size_mb, days_old, days_unaccessed, duplicates]],
                            columns=["file_size_mb", "days_since_modified", "days_since_accessed", "duplicate_count"]
                        )

                        # Prediction
                        pred = classification_model.predict(features.values)[0] if classification_model else 0
                        reason = "Flagged by ML model based on feature pattern."

                        # Conditions for flagging
                        if duplicates > 0:
                            reason = f"Duplicated {duplicates} times."
                        elif size_mb > 100 and days_unaccessed > 90:
                            reason = "Large & Unaccessed > 90 days."
                        elif days_unaccessed > 180:
                            reason = "Unaccessed > 6 months."
                        elif size_mb > 150:
                            reason = "Exceeds 150MB size limit."

                        file_info = {
                            "File Name": filename,
                            "Size (MB)": f"{size_mb:.2f}",
                            "Reason": reason,
                            "Path": full_path
                        }

                        if pred == 2:
                            delete_list.append(file_info)
                        elif pred == 1:
                            review_list.append(file_info)
                        else:
                            keep_count += 1
                    except:
                        continue

            # Dashboard overview
            st.markdown("<h3 style='color:#2b6cb0;'>Live Directory Overview</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Active Files (KEEP)", keep_count)
            col2.metric("Flagged for Review", len(review_list))
            col3.metric("Marked for Deletion", len(delete_list))

            # Bar chart for classification
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = ["KEEP", "REVIEW", "CLEANUP"]
            counts = [keep_count, len(review_list), len(delete_list)]
            colors = ["#63b3ed", "#4299e1", "#2b6cb0"]
            ax.bar(labels, counts, color=colors, width=0.6)
            ax.set_title("File Classification Distribution", fontsize=14, fontweight='bold', color='#2b6cb0')
            ax.set_ylabel("Count", fontsize=12, color='#4a5568')
            for i, count in enumerate(counts):
                ax.text(i, count + 0.5, str(count), ha='center', fontsize=11, fontweight='bold', color='#2d3748')
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            st.pyplot(fig)

            # Files to delete
            st.markdown("<h5 style='color:#2b6cb0;'>Files to Delete (Cleanup)</h5>", unsafe_allow_html=True)
            if delete_list:
                st.dataframe(pd.DataFrame(delete_list))
            else:
                st.markdown("<p style='color:#4a5568;'>No files recommended for deletion.</p>", unsafe_allow_html=True)

            # Files for review
            st.markdown("<h5 style='color:#2b6cb0;'>Files for Review</h5>", unsafe_allow_html=True)
            if review_list:
                st.dataframe(pd.DataFrame(review_list))
            else:
                st.markdown("<p style='color:#4a5568;'>No files flagged for review.</p>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

# Storage Forecasting
st.header("System Capacity Forecasting")
with st.container():
    col_total, col_used, col_free = st.columns(3)
    col_total.metric("Total Capacity", f"{total_gb} GB")
    col_used.metric("Used Storage", f"{used_gb} GB", delta=f"{round((used_gb/total_gb)*100, 1)}%")
    col_free.metric("Free Space Buffer", f"{free_gb} GB", delta=f"{round((free_gb/total_gb)*100, 1)}%")

# Storage usage history
history_path = "storage_history.csv"
if os.path.exists(history_path):
    history_df = pd.read_csv(history_path)
    # Parse date with errors='coerce' to handle invalid data
    history_df["month"] = pd.to_datetime(history_df["month"], errors='coerce')
    # Drop invalid dates
    history_df = history_df.dropna(subset=["month"])
    # Append current month data
    new_row = pd.DataFrame({"month": [current_month], "used_gb": [used_gb]})
    history_df = pd.concat([history_df, new_row], ignore_index=True)
else:
    history_df = pd.DataFrame({"month": [current_month], "used_gb": [used_gb]})

# Remove duplicates & sort
history_df["month"] = pd.to_datetime(history_df["month"], errors='coerce')
history_df = history_df.dropna(subset=["month"])
history_df = history_df.drop_duplicates(subset="month", keep="last")
history_df = history_df.sort_values("month")
history_df["used_gb"] = pd.to_numeric(history_df["used_gb"])
history_df["month_number"] = np.arange(len(history_df))
history_df.to_csv(history_path, index=False)

# Forecasting with linear regression
if len(history_df) >= 3:
    X = history_df[["month_number"]]
    y = history_df["used_gb"]
    model = LinearRegression()
    model.fit(X, y)
    growth_rate = model.coef_[0]
    last_month_num = history_df["month_number"].max()
    next_month_num = np.array([[last_month_num + 1]])
    predicted_gb = model.predict(next_month_num)[0]

    if growth_rate > 0:
        months_left = free_gb / growth_rate
        if months_left > 12:
            years_left = months_left / 12
            warning_msg = f"Growth Rate: +{round(growth_rate, 2)} GB/month. Approx. {round(years_left, 1)} years left."
        else:
            warning_msg = f"Growth Rate: +{round(growth_rate, 2)} GB/month. Approx. {round(months_left, 1)} months left."
    else:
        warning_msg = "Storage usage stable or decreasing; no capacity warning."

    st.markdown(f"<p style='color:#3182ce; font-weight:bold;'>{warning_msg}</p>", unsafe_allow_html=True)

    # Plotting forecast
    months = list(history_df["month"].dt.strftime("%b")) + ["Next"]
    storage_vals = list(history_df["used_gb"]) + [round(predicted_gb, 2)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months[:-1], storage_vals[:-1], marker='o', color='#4299e1', linewidth=2)
    ax.plot(months[-2:], storage_vals[-2:], marker='^', linestyle='--', color='#2b6cb0', linewidth=2)
    for i, val in enumerate(storage_vals):
        ax.text(i, val + max(storage_vals)*0.02, f"{val:.1f}", ha='center', fontsize=9, fontweight='bold', color='#2d3748')
    ax.set_xlabel("Month", fontsize=11, fontweight='bold', color='#4a5568')
    ax.set_ylabel("Storage Usage (GB)", fontsize=11, fontweight='bold', color='#4a5568')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(["Historical", "Forecast"], loc='upper left', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
else:
    st.markdown("<p style='color:#e53e3e;'>Insufficient data for forecasting. Please run over at least 3 months.</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)