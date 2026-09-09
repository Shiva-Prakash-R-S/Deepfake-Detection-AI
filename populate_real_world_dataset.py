import os
import glob
import shutil
import hashlib
import urllib.request
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_WORLD_DIR = os.path.join(PROJECT_ROOT, "dataset", "real_world_test")
FAKE_DIR = os.path.join(REAL_WORLD_DIR, "fake")
REAL_DIR = os.path.join(REAL_WORLD_DIR, "real")

os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

print("Setting up dataset/real_world_test/ ...")

# 1. Collect and copy Gemini generated AI portraits
artifact_dir = r"C:\Users\shiva\.gemini\antigravity-ide\brain\d8144557-850c-463d-afe7-81985b26d793"
gemini_images = [
    ("ai_gen_portrait_male_1788529934723.jpg", "gemini_ai_portrait_male.jpg"),
    ("ai_gen_portrait_female_1788529963218.jpg", "gemini_ai_portrait_female.jpg"),
    ("ai_gen_elderly_man_1788529999645.jpg", "gemini_ai_portrait_elderly_man.jpg"),
]

for src_name, dest_name in gemini_images:
    src_path = os.path.join(artifact_dir, src_name)
    if os.path.exists(src_path):
        dest_path = os.path.join(FAKE_DIR, dest_name)
        shutil.copyfile(src_path, dest_path)
        print(f"Added Gemini AI Image: {dest_name}")

# 2. Collect unique WhatsApp/Midjourney/SD fake images from real_vs_fake_realimg
seen_fake_hashes = set()
fake_count = 0
for f in sorted(glob.glob(os.path.join(PROJECT_ROOT, "dataset", "real_vs_fake_realimg", "**", "fake", "*.*"), recursive=True)):
    if os.path.isfile(f):
        with open(f, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        if h not in seen_fake_hashes:
            seen_fake_hashes.add(h)
            fake_count += 1
            ext = os.path.splitext(f)[1].lower()
            if not ext or ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"
            dest_name = f"ai_generated_img_{fake_count:02d}{ext}"
            dest_path = os.path.join(FAKE_DIR, dest_name)
            # Ensure saved as clean RGB image
            try:
                im = Image.open(f).convert("RGB")
                im.save(dest_path)
            except Exception as e:
                shutil.copyfile(f, dest_path)

print(f"Added {fake_count} unseen AI-generated images to {FAKE_DIR}")

# 3. Collect unique mobile real photos from real_vs_fake_realimg
seen_real_hashes = set()
real_count = 0
for f in sorted(glob.glob(os.path.join(PROJECT_ROOT, "dataset", "real_vs_fake_realimg", "**", "real", "*.*"), recursive=True)):
    if os.path.isfile(f):
        with open(f, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()
        if h not in seen_real_hashes:
            seen_real_hashes.add(h)
            real_count += 1
            dest_name = f"real_photo_mobile_{real_count:02d}.jpg"
            dest_path = os.path.join(REAL_DIR, dest_name)
            try:
                im = Image.open(f).convert("RGB")
                im.save(dest_path)
            except Exception:
                shutil.copyfile(f, dest_path)
            if real_count >= 24:
                break

print(f"Added {real_count} unseen mobile real photos to {REAL_DIR}")

# 4. Download 5 additional genuine real-world studio/outdoor DSLR portraits from Unsplash
unsplash_urls = [
    ("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&q=85", "real_photo_dslr_01.jpg"),
    ("https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=600&q=85", "real_photo_dslr_02.jpg"),
    ("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600&q=85", "real_photo_dslr_03.jpg"),
    ("https://images.unsplash.com/photo-1517841905240-472988babdf9?w=600&q=85", "real_photo_dslr_04.jpg"),
    ("https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=600&q=85", "real_photo_dslr_05.jpg"),
]

for url, fname in unsplash_urls:
    dest_path = os.path.join(REAL_DIR, fname)
    if not os.path.exists(dest_path):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            with open(dest_path, "wb") as fp:
                fp.write(data)
            print(f"Downloaded DSLR real photo: {fname}")
        except Exception as e:
            print(f"Failed downloading {fname}: {e}")

final_fakes = len(glob.glob(os.path.join(FAKE_DIR, "*.*")))
final_reals = len(glob.glob(os.path.join(REAL_DIR, "*.*")))

print("\n" + "=" * 50)
print(f"Dataset summary in {REAL_WORLD_DIR}:")
print(f"  Total Fake (AI-generated) images: {final_fakes}")
print(f"  Total Real (genuine photographs) images: {final_reals}")
print(f"  Total evaluation dataset: {final_fakes + final_reals}")
print("=" * 50)
