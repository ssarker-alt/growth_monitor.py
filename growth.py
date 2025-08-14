# growth_monitor.py
import json
from datetime import datetime
from pathlib import Path
import cv2  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image  # type: ignore
import pillow_heif  # type: ignore
import streamlit as st  # type: ignore
import shutil

# --- Paths ---
DATA_DIR = Path("growth_data")
SAMPLES_FILE = DATA_DIR / "samples.json"
OBS_FILE = DATA_DIR / "observations.csv"
IMAGES_DIR = DATA_DIR / "images"
for p in [DATA_DIR, IMAGES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

pillow_heif.register_heif_opener()

# --- Data Helpers ---
def load_samples():
    return json.loads(SAMPLES_FILE.read_text()) if SAMPLES_FILE.exists() else {}

def save_samples(samples):
    SAMPLES_FILE.write_text(json.dumps(samples, indent=2))

def load_observations():
    if OBS_FILE.exists():
        return pd.read_csv(OBS_FILE, parse_dates=["timestamp"])
    cols = ["timestamp", "sample_id", "image_path",
            "coverage_pct", "green_pixels", "brown_pixels",
            "humidity", "light_exposure", "notes"]
    return pd.DataFrame(columns=cols)

def save_observation(row):
    df = load_observations()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(OBS_FILE, index=False)

# --- Image Helpers ---
def ensure_sample_folder(sample_id):
    folder = IMAGES_DIR / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def save_image(pil_img, sample_id):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = ensure_sample_folder(sample_id) / f"{sample_id}__{ts}.jpg"
    pil_img.save(path, "JPEG", quality=90)
    return str(path)

def detect_growth(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    brown_mask = cv2.inRange(hsv, (10, 50, 20), (30, 255, 200))
    green_count = int(np.sum(green_mask > 0))
    brown_count = int(np.sum(brown_mask > 0))
    combined = cv2.bitwise_or(green_mask, brown_mask)
    coverage = (np.sum(combined > 0) / combined.size) * 100 if combined.size > 0 else 0
    highlighted = img_cv.copy()
    highlighted[combined > 0] = [255, 255, 255]
    return {
        "coverage_pct": round(coverage, 2),
        "green_pixels": green_count,
        "brown_pixels": brown_count,
        "highlighted_img": Image.fromarray(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    }

def compute_health(coverage_pct, green_pixels, brown_pixels, change):
    cov_score = min(max(coverage_pct, 0), 100) * 0.5
    total = green_pixels + brown_pixels
    green_ratio = green_pixels / total if total > 0 else 0.5
    gr_score = green_ratio * 30
    change_score = 10 + (max(min(change or 0, 20), -20) / 20) * 10
    return round(min(max(cov_score + gr_score + change_score, 0), 100), 2), round(green_ratio, 2)

# --- Streamlit UI ---
st.set_page_config(page_title="🌿 Growth Monitor", layout="wide")
st.title("🌿 Organism Growth Monitor")

# Session state for refresh
if "refresh" not in st.session_state:
    st.session_state.refresh = False

samples = load_samples()

# Sidebar: Manage Samples
with st.sidebar:
    st.header("Sample Management")
    sid = st.text_input("New Sample ID")
    species = st.text_input("Species / Notes")
    if st.button("Create Sample"):
        if sid and sid not in samples:
            samples[sid] = {"species": species, "start_date": datetime.utcnow().date().isoformat()}
            save_samples(samples)
            st.success(f"Created {sid}")
            st.session_state.refresh = True
        else:
            st.error("Invalid or duplicate Sample ID")
    st.write("---")
    for s_id in list(samples.keys()):
        st.write(f"**{s_id}** — {samples[s_id]['species']}")
        if st.button(f"Delete {s_id}", key=f"del_{s_id}"):
            confirm = st.radio(
                f"Confirm delete {s_id}?",
                options=["No", "Yes"],
                key=f"confirm_{s_id}",
                horizontal=True
            )
            if confirm == "Yes":
                del samples[s_id]
                save_samples(samples)
                st.success(f"Deleted {s_id}")
                st.session_state.refresh = True

if st.session_state.refresh:
    st.session_state.refresh = False
    st.experimental_rerun()

# Left: Add Observation
left, right = st.columns([1.3, 1])
with left:
    st.header("📸 Add Observation")
    sample_choice = st.selectbox("Sample", list(samples.keys()) or ["(none)"])
    src = st.radio("Image Source", ["Upload", "Camera"])
    image = None
    if src == "Upload":
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "heic"])
        if file:
            image = Image.open(file).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    humidity = st.number_input("Humidity (%)", 0, 100, 50)
    light = st.selectbox("Light Exposure", ["Low", "Medium", "High"])
    notes = st.text_area("Notes")

    if image:
        stats = detect_growth(image)
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(stats["highlighted_img"], caption="Detected Growth", use_container_width=True)
        st.metric("Coverage (%)", stats["coverage_pct"])

        if st.button("Save Observation"):
            if sample_choice not in samples:
                samples[sample_choice] = {"species": "", "start_date": datetime.utcnow().date().isoformat()}
                save_samples(samples)
            img_path = save_image(image, sample_choice)
            save_observation({
                "timestamp": datetime.utcnow().isoformat(),
                "sample_id": sample_choice,
                "image_path": img_path,
                "coverage_pct": stats["coverage_pct"],
                "green_pixels": stats["green_pixels"],
                "brown_pixels": stats["brown_pixels"],
                "humidity": humidity,
                "light_exposure": light,
                "notes": notes
            })
            st.success("Observation saved.")
            st.session_state.refresh = True

# Right: Dashboard
with right:
    st.header("📊 Dashboard")
    obs_df = load_observations()

    if obs_df.empty:
        st.info("No observations yet.")
    else:
        last = obs_df.sort_values("timestamp").groupby("sample_id").last().reset_index()
        changes, scores, ratios = [], [], []
        for sid in last["sample_id"]:
            s_obs = obs_df[obs_df["sample_id"] == sid].sort_values("timestamp")
            change = s_obs.iloc[-1]["coverage_pct"] - s_obs.iloc[-2]["coverage_pct"] if len(s_obs) > 1 else None
            score, ratio = compute_health(
                s_obs.iloc[-1]["coverage_pct"],
                s_obs.iloc[-1]["green_pixels"],
                s_obs.iloc[-1]["brown_pixels"],
                change
            )
            changes.append(change)
            scores.append(score)
            ratios.append(ratio)
        last["Δ since prev"] = changes
        last["HEALTH"] = scores
        last["Green Ratio"] = ratios

        # Show table
        st.dataframe(last.sort_values("HEALTH", ascending=False))

        # Select a sample to view/manage
        sample_view = st.selectbox("View or manage sample", last["sample_id"])
        if sample_view:
            s_obs = obs_df[obs_df["sample_id"] == sample_view].sort_values("timestamp")
            st.line_chart(s_obs.set_index("timestamp")["coverage_pct"])

            # Delete sample
            if st.button(f"❌ Delete sample {sample_view} and all its data", key=f"del_sample_{sample_view}"):
                if sample_view in samples:
                    del samples[sample_view]
                    save_samples(samples)
                obs_df = obs_df[obs_df["sample_id"] != sample_view]
                obs_df.to_csv(OBS_FILE, index=False)
                sample_folder = IMAGES_DIR / sample_view
                if sample_folder.exists():
                    shutil.rmtree(sample_folder)
                st.success(f"Deleted sample {sample_view} and all


