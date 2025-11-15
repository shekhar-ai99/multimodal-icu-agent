"""Generate synthetic MIMIC-III-like data for 100 ICU stays.

Creates:
 - data/mimic3_dummy.csv
 - data/cxr_dummy/*.png
 - data/metadata.json (train/val/test split by subject_id)

Usage: python src/data_loader.py
"""
import os
import json
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _make_cxr_image(path, text, size=(512, 512), seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img = Image.new("L", size, color=40)
    draw = ImageDraw.Draw(img)
    # Add some noisy circles/ellipses to mimic lung fields
    for _ in range(30):
        x0 = random.randint(0, size[0])
        y0 = random.randint(0, size[1])
        x1 = x0 + random.randint(10, 120)
        y1 = y0 + random.randint(10, 120)
        shade = random.randint(30, 120)
        draw.ellipse([x0, y0, x1, y1], outline=shade)
    # Draw the label text in a corner
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, size[1] - 20), text, fill=220, font=font)
    img.save(path)


def generate_mimic_dummy(output_dir="data", n_patients=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    cxr_dir = os.path.join(output_dir, "cxr_dummy")
    os.makedirs(cxr_dir, exist_ok=True)

    records = []
    metadata = {"subjects": [], "splits": {}}

    # Create 100 dummy CXR images with a mix of labels
    cxr_texts = [
        "Normal",
        "Bilateral opacities",
        "Consolidation RLL",
        "Pleural effusion",
        "Mass",
        "Cardiomegaly",
    ]
    cxr_paths = []
    for i in range(n_patients):
        text = random.choice(cxr_texts)
        fname = f"cxr_{i:03d}.png"
        path = os.path.join(cxr_dir, fname)
        _make_cxr_image(path, text)
        cxr_paths.append(path)

    # diagnoses and prevalences
    diagnoses = ["pneumonia", "sepsis", "heart_failure", "other"]

    for pid in range(n_patients):
        subject_id = 10000 + pid
        hadm_id = 50000 + pid
        icu_stay_id = 70000 + pid

        # 24 hours at 1-min intervals
        start_time = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 365))
        times = [start_time + timedelta(minutes=i) for i in range(1440)]

        # baseline vitals
        hr_base = int(np.random.normal(80, 8))
        sys_base = int(np.random.normal(120, 10))
        dias_base = int(np.random.normal(70, 7))
        rr_base = int(np.random.normal(16, 2))
        spo2_base = int(np.random.normal(97, 1))
        temp_base = float(np.random.normal(36.8, 0.2))

        # assign diagnosis and whether shock occurs
        diagnosis = random.choices(diagnoses, weights=[0.25, 0.2, 0.15, 0.4])[0]
        label_shock = int(random.random() < 0.18)  # ~18% shock prevalence

        # pick a cxr for this patient
        cxr_path = os.path.relpath(cxr_paths[pid], output_dir)

        # notes: 3-5 per stay at random times
        n_notes = random.randint(3, 5)
        note_times = sorted(random.sample(range(1440), n_notes))
        notes = []
        for nt in note_times:
            note = "Progress note: "
            if diagnosis == "pneumonia" and random.random() < 0.7:
                note += random.choice([
                    "Bilateral opacities on CXR",
                    "Consolidation noted in RLL",
                    "Suspected pneumonia; starting antibiotics",
                ])
            elif label_shock and random.random() < 0.6:
                note += random.choice([
                    "Hypotension refractory to fluids",
                    "Vasopressor started for shock",
                    "Lactate elevated; concern for shock",
                ])
            else:
                note += random.choice([
                    "Patient stable",
                    "Improving clinically",
                    "Plan: monitor vitals closely",
                ])
            notes.append((nt, note))

        # simulate vitals per minute
        hr = np.clip(np.random.normal(hr_base, 3, 1440).astype(int), 35, 220)
        sysbp = np.clip(np.random.normal(sys_base, 6, 1440).astype(int), 40, 260)
        diasbp = np.clip(np.random.normal(dias_base, 4, 1440).astype(int), 20, 160)
        rr = np.clip(np.random.normal(rr_base, 1.5, 1440).astype(int), 6, 60)
        spo2 = np.clip(np.random.normal(spo2_base, 1.0, 1440).astype(int), 60, 100)
        temp = np.clip(np.random.normal(temp_base, 0.15, 1440), 34.0, 41.0)

        # if pneumonia: increase temp and rr for some window
        if diagnosis == "pneumonia":
            fever_start = random.randint(100, 800)
            fever_end = min(1439, fever_start + random.randint(120, 600))
            temp[fever_start:fever_end] += np.random.normal(1.0, 0.3, fever_end - fever_start)
            rr[fever_start:fever_end] += np.random.poisson(3, fever_end - fever_start)

        # if shock occurs: change vitals starting 6h before the last timepoint (label window)
        if label_shock:
            onset_min = 1440 - (6 * 60)  # 6 hours before end
            hr[onset_min:] += np.random.randint(15, 35)
            sysbp[onset_min:] -= np.random.randint(20, 40)
            diasbp[onset_min:] -= np.random.randint(10, 25)
            spo2[onset_min:] -= np.random.randint(1, 6)
            temp[onset_min:] += np.random.normal(0.3, 0.2, 1440 - onset_min)

        # Build rows
        note_index = {nt: txt for (nt, txt) in notes}
        for i, ct in enumerate(times):
            note_text = note_index.get(i, "")
            records.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "icu_stay_id": icu_stay_id,
                    "charttime": ct.isoformat(sep=" "),
                    "hr": int(hr[i]),
                    "sysbp": int(sysbp[i]),
                    "diasbp": int(diasbp[i]),
                    "resp_rate": int(rr[i]),
                    "spo2": int(spo2[i]),
                    "temp": float(round(float(temp[i]), 2)),
                    "note_text": note_text,
                    "cxr_path": cxr_path,
                    "label_shock": int(label_shock),
                    "diagnosis": diagnosis,
                }
            )

        metadata["subjects"].append(subject_id)

    # Create train/val/test split by subject
    subjects = metadata["subjects"]
    random.shuffle(subjects)
    n = len(subjects)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = subjects[:n_train]
    val = subjects[n_train : n_train + n_val]
    test = subjects[n_train + n_val :]
    metadata["splits"] = {"train": train, "val": val, "test": test}

    # Save CSV
    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(output_dir, "mimic3_dummy.csv")
    df.to_csv(csv_path, index=False)

    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Created {len(cxr_paths)} CXR images in {cxr_dir}")


if __name__ == "__main__":
    generate_mimic_dummy(output_dir="data", n_patients=100, seed=42)
