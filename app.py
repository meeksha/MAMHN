# ============================================================
# Memory-Augmented Meta-Hypernetwork for Adaptive Anomaly Detection
# Final Streamlit App - Corrected and Demo-Stable
# ============================================================

import os, warnings
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"   # prevent torch.classes scanning
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time

# ---------------- CONFIG ----------------
OUT_DIR = r"C:\Users\1011s\Desktop\mini\Processed_IoT_dataset\weather_preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

scaler_path = os.path.join(OUT_DIR, "minmax_scaler.joblib")
if not os.path.exists(scaler_path):
    st.error(f"Scaler not found at {scaler_path}. Place minmax_scaler.joblib in OUT_DIR.")
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ------------------- Model Definitions ----------------------
# ============================================================
class MemoryModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class MetaHyperNetwork(nn.Module):
    def __init__(self, mem_dim, main_in, main_hidden):
        super().__init__()
        self.fc1 = nn.Linear(mem_dim, 64)
        self.fc2 = nn.Linear(64, main_hidden * main_in + main_hidden)
    def forward(self, mem):
        x = torch.relu(self.fc1(mem))
        return self.fc2(x)

class MainNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x, params=None):
        if params is not None:
            total_w = self.hidden_dim * self.in_dim
            w = params[:, :total_w].view(-1, self.hidden_dim, self.in_dim)
            b = params[:, total_w:].view(-1, self.hidden_dim)
            x_exp = x.unsqueeze(2)
            lin = torch.bmm(w, x_exp).squeeze(2) + b
            h = torch.relu(lin)
        else:
            h = torch.relu(self.linear(x))
        return torch.sigmoid(self.out(h))

# ============================================================
# ------------------- Streamlit Setup ------------------------
# ============================================================
st.set_page_config(page_title="IoT Anomaly Detection Dashboard", layout="wide")
st.title(" Memory-Augmented Meta-Hypernetwork for Adaptive Anomaly Detection")

st.sidebar.header("Configuration")
window_size = st.sidebar.slider("Sliding Window Size", 1, 20, 10)
threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, format="%.6f")
use_v2 = st.sidebar.checkbox("Use retrained model (v2)", value=True)

# ---------------- load scaler and infer feature names ----------------
scaler = joblib.load(scaler_path)
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    n_feats = getattr(scaler, "n_features_in_", None)
    if n_feats is None:
        st.error("Scaler does not expose feature names or n_features_in_. Provide a compatible scaler.")
        st.stop()
    feature_names = [f"f{i}" for i in range(n_feats)]

input_dim = len(feature_names)
hidden_dim = 32

# ---------------- instantiate models (must be before loading weights) --------------
memory = MemoryModule(input_dim, hidden_dim).to(DEVICE)
meta = MetaHyperNetwork(hidden_dim, input_dim, hidden_dim).to(DEVICE)
main = MainNetwork(input_dim, hidden_dim).to(DEVICE)

# ---------------- load selected weights ----------------
def safe_load_state(model, path):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            return True, None
        except Exception as e:
            return False, str(e)
    return False, f"File not found: {path}"

if use_v2:
    ok, err = safe_load_state(memory, os.path.join(OUT_DIR, "memory_v2.pt"))
    if not ok:
        st.warning(f"Could not load memory_v2.pt: {err} — falling back to memory.pt")
        safe_load_state(memory, os.path.join(OUT_DIR, "memory.pt"))
    ok, err = safe_load_state(meta, os.path.join(OUT_DIR, "meta_v2.pt"))
    if not ok:
        st.warning(f"Could not load meta_v2.pt: {err} — falling back to meta.pt")
        safe_load_state(meta, os.path.join(OUT_DIR, "meta.pt"))
    ok, err = safe_load_state(main, os.path.join(OUT_DIR, "main_v2.pt"))
    if not ok:
        st.warning(f"Could not load main_v2.pt: {err} — falling back to main.pt")
        safe_load_state(main, os.path.join(OUT_DIR, "main.pt"))
else:
    safe_load_state(memory, os.path.join(OUT_DIR, "memory.pt"))
    safe_load_state(meta, os.path.join(OUT_DIR, "meta.pt"))
    safe_load_state(main, os.path.join(OUT_DIR, "main.pt"))

memory.eval(); meta.eval(); main.eval()

# ============================================================
# ------------------- File Upload -----------------------------
# ============================================================
uploaded_file = st.file_uploader(" Upload IoT Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data Preview")
    st.dataframe(df.head())

    # --- downsample large datasets for demo stability ---
    if len(df) > 5000:
        st.warning(f"Dataset too large ({len(df)} rows). Downsampling to 5000 for smooth performance.")
        df = df.sample(n=5000, random_state=42).sort_index().reset_index(drop=True)

    # --- scale + reorder columns ---
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in input: {missing_cols}. Filling missing with zeros.")
    df = df.reindex(columns=feature_names, fill_value=0)

    # scale (use DataFrame values aligned to feature_names)
    try:
        scaled = scaler.transform(df[feature_names].values)
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.stop()

    # --- sliding windows ---
    if len(scaled) <= window_size:
        st.error("Dataset too small for the chosen window size.")
        st.stop()

    # build windows in a memory-efficient way (list -> stack)
    X_list = [scaled[i:i + window_size] for i in range(len(scaled) - window_size + 1)]
    X_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32).to(DEVICE)
    st.info(f" Created {len(X_tensor)} windows (each with {window_size} timesteps).")

    # ============================================================
    # ------------------- Model Inference ------------------------
    # ============================================================
    with torch.no_grad():
        mem = memory(X_tensor)
        params = meta(mem)
        preds = main(X_tensor[:, -1, :], params).cpu().numpy().reshape(-1)

    # --- normalize for human-readable scale (0–1) ---
    preds = preds - preds.min()
    if preds.max() > 0:
        preds = preds / preds.max()

    anomaly_flags = (preds > threshold).astype(int)

    # --- build results dataframe ---
    df_result = df.iloc[window_size - 1:].copy().reset_index(drop=True)
    df_result["Anomaly_Prob"] = preds
    df_result["Predicted_Label"] = anomaly_flags

    # ============================================================
    # ------------------- Visualization --------------------------
    # ============================================================
    st.subheader(" Detection Results (Last 10 Rows)")
    st.dataframe(df_result.tail(10))

    normal_count = int((df_result["Predicted_Label"] == 0).sum())
    anomaly_count = int((df_result["Predicted_Label"] == 1).sum())
    total = len(df_result)
    anomaly_rate = (anomaly_count / total) * 100 if total > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Normal Windows", normal_count)
    col2.metric("Anomaly Windows", anomaly_count)
    col3.metric("Anomaly Rate (%)", f"{anomaly_rate:.2f}")

    # charting - avoid rendering massive charts
    if len(df_result) < 20000:
        st.line_chart(df_result["Anomaly_Prob"], height=250, use_container_width=True)
    else:
        st.info("Dataset too large to render chart; showing summary metrics only.")

    st.bar_chart(df_result["Predicted_Label"].value_counts())

    # --- download results ---
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Results CSV", data=csv, file_name="anomaly_results.csv")

else:
    st.info(" Upload a CSV file to start detection.")
