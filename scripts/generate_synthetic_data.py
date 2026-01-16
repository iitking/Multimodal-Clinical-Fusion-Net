"""
Synthetic Clinical Dataset Generator for Multimodal-Clinical-Fusion-Net.

Generates realistic tabular EHR records and paired synthetic Chest X-ray images.
Tabular features include demographics, vital signs, and laboratory values.
Paired via: subject_id (Patient ID) and study_id (Imaging Study Encounter ID).
"""

import os
import csv
import math
import random
import zlib
import struct
from pathlib import Path


def generate_pure_python_png(width: int, height: int, pixels: list) -> bytes:
    """
    Encodes an 8-bit grayscale image as a standard valid PNG in pure Python.
    `pixels` is a flat list of length width * height with values [0, 255].
    """
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # filter type 0 (None)
        row_start = y * width
        raw_data.extend(pixels[row_start : row_start + width])

    compressed = zlib.compress(bytes(raw_data), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    png_header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=0 (grayscale), compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")

    return png_header + ihdr + idat + iend


def draw_synthetic_chest_xray(width: int, height: int, has_abnormality: bool = False) -> list:
    """
    Simulates a frontal Chest X-ray grayscale image:
    - Dark background with soft thoracic outline
    - Bilateral darker lung fields (air is radiolucent / dark)
    - Bright mediastinum, spine, and ribs (bone/tissue are radiopaque / bright)
    - If has_abnormality is True: creates focal opacity/infiltrate (e.g. pneumonia or consolidation)
    """
    pixels = [0] * (width * height)
    cx, cy = width / 2.0, height / 2.0

    # Anatomic centers for left and right lung fields
    left_lung_cx, left_lung_cy = width * 0.33, height * 0.52
    right_lung_cx, right_lung_cy = width * 0.67, height * 0.52
    lung_rx, lung_ry = width * 0.16, height * 0.28

    for y in range(height):
        ny = (y - cy) / (height / 2.0)
        for x in range(width):
            nx = (x - cx) / (width / 2.0)
            idx = y * width + x

            # Outer body contour (oval gradient)
            r_body = math.sqrt(nx * nx + ny * ny)
            if r_body > 0.95:
                val = 20 + random.randint(-3, 3)
            else:
                # Thoracic tissue base brightness
                val = 140 - int(r_body * 50)

                # Central mediastinum & cardiac silhouette (spine/heart is brighter)
                spine_dist = abs(nx)
                if spine_dist < 0.14 and abs(ny) < 0.8:
                    val += 50
                # Heart shadow on patient's left (viewer's right, lower lung)
                cardiac_dist = math.sqrt((nx - 0.12) ** 2 + (ny - 0.2) ** 2)
                if cardiac_dist < 0.28:
                    val += 60

                # Left lung field (darker radiolucent area)
                dl = ((x - left_lung_cx) / lung_rx) ** 2 + ((y - left_lung_cy) / lung_ry) ** 2
                if dl < 1.0:
                    val -= int((1.0 - dl) * 75)

                # Right lung field
                dr = ((x - right_lung_cx) / lung_rx) ** 2 + ((y - right_lung_cy) / lung_ry) ** 2
                if dr < 1.0:
                    val -= int((1.0 - dr) * 75)

                # Horizontal rib shadows
                rib_wave = math.sin((y / height) * 26.0)
                if dl < 1.1 or dr < 1.1:
                    val += int(rib_wave * 12)

                # If clinical abnormality is present: simulate focal pulmonary opacity / consolidation
                if has_abnormality:
                    # Focal infiltrate in right lower zone
                    focal_d = math.sqrt(((x - right_lung_cx - 5) / 22.0) ** 2 + ((y - right_lung_cy - 15) / 20.0) ** 2)
                    if focal_d < 1.2:
                        val += int((1.2 - focal_d) * 85)

            # Natural X-ray quantum noise
            val += random.randint(-4, 4)
            pixels[idx] = max(0, min(255, val))

    return pixels


def generate_synthetic_dataset(num_patients: int = 200, output_dir: str = "."):
    """
    Generates tabular records and paired chest X-rays.
    Patients have unique `subject_id`.
    Each patient can have 1 to 2 imaging encounters (`study_id`).
    """
    random.seed(42)
    base_path = Path(output_dir)
    tabular_dir = base_path / "data" / "raw" / "tabular"
    images_dir = base_path / "data" / "raw" / "images"
    tabular_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    tabular_records = []
    study_counter = 500000

    print(f"Generating synthetic clinical data for {num_patients} patients...")

    for i in range(1, num_patients + 1):
        subject_id = f"SUBJ_{10000 + i}"
        age = random.randint(22, 88)
        gender = random.choice(["M", "F"])

        # Decide baseline health condition / risk
        # Older patients with higher baseline risk
        patient_risk_latent = (age / 100.0) * 0.4 + (0.3 if random.random() > 0.65 else 0.0)

        # Number of studies for this patient
        num_studies = 1 if random.random() < 0.75 else 2

        for _ in range(num_studies):
            study_counter += 1
            study_id = f"STUDY_{study_counter}"

            # Encounter-level health shock
            encounter_risk = min(0.95, patient_risk_latent + random.uniform(-0.15, 0.35))
            is_critical = 1 if encounter_risk > 0.55 else 0

            # Realistic lab values conditioned on clinical state
            if is_critical:
                heart_rate = round(random.gauss(108, 14), 1)        # Tachycardia
                sbp = round(random.gauss(92, 12), 1)                # Hypotension
                dbp = round(random.gauss(58, 8), 1)
                resp_rate = round(random.gauss(26, 4), 1)          # Tachypnea
                spo2 = round(min(96.0, random.gauss(88.0, 3.5)), 1) # Hypoxemia
                temp = round(random.gauss(38.4, 0.7), 1)            # Fever
                wbc = round(random.gauss(16.5, 4.0), 1)             # Leukocytosis
                creatinine = round(random.gauss(2.4, 0.8), 2)       # Acute Kidney Injury
                glucose = round(random.gauss(175, 45), 1)           # Hyperglycemia
                hemoglobin = round(random.gauss(10.2, 1.8), 1)      # Anemia
                platelets = round(random.gauss(140, 45), 0)         # Thrombocytopenia
                has_xray_opacity = (random.random() < 0.85)
            else:
                heart_rate = round(random.gauss(76, 9), 1)
                sbp = round(random.gauss(122, 10), 1)
                dbp = round(random.gauss(78, 6), 1)
                resp_rate = round(random.gauss(16, 2), 1)
                spo2 = round(min(100.0, random.gauss(98.2, 1.1)), 1)
                temp = round(random.gauss(36.8, 0.3), 1)
                wbc = round(random.gauss(7.2, 1.5), 1)
                creatinine = round(random.gauss(0.9, 0.2), 2)
                glucose = round(random.gauss(105, 18), 1)
                hemoglobin = round(random.gauss(14.0, 1.2), 1)
                platelets = round(random.gauss(250, 40), 0)
                has_xray_opacity = (random.random() < 0.15)

            # Render synthetic CXR (224x224)
            img_width, img_height = 224, 224
            pixels = draw_synthetic_chest_xray(img_width, img_height, has_abnormality=has_xray_opacity)
            png_bytes = generate_pure_python_png(img_width, img_height, pixels)

            image_filename = f"{study_id}.png"
            image_filepath = images_dir / image_filename
            with open(image_filepath, "wb") as f_img:
                f_img.write(png_bytes)

            record = {
                "subject_id": subject_id,
                "study_id": study_id,
                "image_path": str(Path("data/raw/images") / image_filename),
                "age": age,
                "gender": gender,
                "heart_rate": heart_rate,
                "sbp": sbp,
                "dbp": dbp,
                "resp_rate": resp_rate,
                "spo2": spo2,
                "temp": temp,
                "wbc": wbc,
                "creatinine": creatinine,
                "glucose": glucose,
                "hemoglobin": hemoglobin,
                "platelets": platelets,
                "outcome": is_critical  # 1: High clinical risk / adverse outcome, 0: Stable
            }
            tabular_records.append(record)

    # Save CSV
    csv_path = tabular_dir / "patient_records.csv"
    fieldnames = list(tabular_records[0].keys())
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tabular_records)

    print(f"Generated {len(tabular_records)} paired multimodal records.")
    print(f"Tabular CSV saved at: {csv_path}")
    print(f"Images saved at: {images_dir}")


if __name__ == "__main__":
    generate_synthetic_dataset(num_patients=180, output_dir=".")
