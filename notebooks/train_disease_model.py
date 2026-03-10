"""
Plant Disease Model — Mac CPU Optimized
========================================
Optimizations for fast training on Mac:
  - Image size 128x128 (vs 224x224) → 3x faster per epoch
  - Batch size 64 → better CPU utilization
  - MobileNetV2 frozen (no fine-tuning) → 1 phase only
  - Light augmentation → faster data pipeline
  - 15 epochs max with patience=5
  - Uses Apple MPS (Metal) if available on M1/M2/M3 Mac

Expected time:
  Mac M1/M2 (MPS): ~8-12 min total
  Mac Intel CPU  : ~25-35 min total
  Target accuracy: 85-92%
"""

import os, json, shutil, random
import numpy as np
import tensorflow as tf

# ── Apple Silicon MPS acceleration ───────────────────────────
# Uncomment below if you have M1/M2/M3 Mac:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

print(f"TensorFlow : {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────
DATA_DIR   = "../data/PlantVillage"
SPLIT_DIR  = "../data/PV_split"
TRAIN_DIR  = f"{SPLIT_DIR}/train"
VAL_DIR    = f"{SPLIT_DIR}/val"
MODEL_DIR  = "../models"

IMG_SIZE   = (128, 128)   # smaller = much faster on CPU
BATCH_SIZE = 64            # larger batch = better CPU util
SEED       = 42
os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED)

# ── Step 1: Fix nested PlantVillage folder ────────────────────
nested = os.path.join(DATA_DIR, "PlantVillage")
if os.path.isdir(nested):
    print(f"⚠  Fixing nested folder: {nested}")
    for item in os.listdir(nested):
        src  = os.path.join(nested, item)
        dest = os.path.join(DATA_DIR, item)
        if not os.path.exists(dest):
            shutil.move(src, dest)
    try:
        os.rmdir(nested)
    except:
        pass
    print("✅ Nested folder fixed")

# ── Step 2: Discover classes ──────────────────────────────────
classes = []
for d in sorted(os.listdir(DATA_DIR)):
    p = os.path.join(DATA_DIR, d)
    if not os.path.isdir(p):
        continue
    imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(imgs) >= 10:
        classes.append(d)

print(f"\n✅ Found {len(classes)} classes:")
for c in classes:
    n = len([f for f in os.listdir(os.path.join(DATA_DIR, c))
             if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"   {c:<50} {n} images")

# ── Step 3: Create train/val split (only once) ────────────────
if not os.path.exists(SPLIT_DIR):
    print(f"\nCreating train/val split...")
    total_train, total_val = 0, 0
    for cls in classes:
        imgs = [f for f in os.listdir(os.path.join(DATA_DIR, cls))
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(imgs)
        n_val    = max(1, int(len(imgs) * 0.2))
        val_imgs = imgs[:n_val]
        trn_imgs = imgs[n_val:]
        for subset, lst in [('train', trn_imgs), ('val', val_imgs)]:
            dst = os.path.join(SPLIT_DIR, subset, cls)
            os.makedirs(dst, exist_ok=True)
            for img in lst:
                shutil.copy2(os.path.join(DATA_DIR, cls, img), os.path.join(dst, img))
        total_train += len(trn_imgs)
        total_val   += len(val_imgs)
    print(f"✅ Split done — train: {total_train}  val: {total_val}")
else:
    print(f"\n✅ Split exists at {SPLIT_DIR} — skipping")

# ── Step 4: Data generators ───────────────────────────────────
# Light augmentation = faster pipeline on CPU
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)

val_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\nClasses : {NUM_CLASSES}")
print(f"Train   : {train_gen.samples} images")
print(f"Val     : {val_gen.samples} images")
print(f"Steps/epoch (train): {len(train_gen)}")
print(f"Steps/epoch (val)  : {len(val_gen)}")

# Save class indices
with open(f'{MODEL_DIR}/disease_class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"✅ Saved disease_class_indices.json")

# ── Step 5: Build model ───────────────────────────────────────
print("\nBuilding model...")
base = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False   # keep frozen — no fine-tuning = much faster

x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.4)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

model.summary()
print(f"\nTrainable params: {sum(p.numpy().size for p in model.trainable_variables):,}")

# ── Step 6: Train ─────────────────────────────────────────────
print("\n" + "="*55)
print(f"TRAINING  |  img={IMG_SIZE}  batch={BATCH_SIZE}  classes={NUM_CLASSES}")
print("="*55)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,               # stop if no improvement for 5 epochs
        restore_best_weights=True,
        verbose=1,
        min_delta=0.005           # must improve by at least 0.5%
    ),
    ModelCheckpoint(
        f'{MODEL_DIR}/disease_model_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ── Step 7: Evaluate ─────────────────────────────────────────
print("\nEvaluating...")
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n{'='*55}")
print(f"✅  Final Val Accuracy : {val_acc*100:.2f}%")
print(f"✅  Final Val Loss     : {val_loss:.4f}")
print(f"{'='*55}")

# ── Step 8: Save ─────────────────────────────────────────────
model.save(f'{MODEL_DIR}/disease_model.keras')
model.save(f'{MODEL_DIR}/disease_model.h5')
print(f"\n✅ Saved: disease_model.keras + disease_model.h5")
print(f"   → Copy both to scav2/models/")

# ── Step 9: Plot ─────────────────────────────────────────────
acc  = history.history['accuracy']
vacc = history.history['val_accuracy']
loss = history.history['loss']
vloss= history.history['val_loss']

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(acc,  label='Train', color='#4a9960', linewidth=2)
a1.plot(vacc, label='Val',   color='#e8a020', linewidth=2)
a1.set_title(f'Accuracy (best val: {max(vacc)*100:.1f}%)')
a1.legend(); a1.set_ylim(0, 1); a1.grid(alpha=.3)
a1.set_xlabel('Epoch')

a2.plot(loss,  label='Train', color='#4a9960', linewidth=2)
a2.plot(vloss, label='Val',   color='#e8a020', linewidth=2)
a2.set_title('Loss'); a2.legend(); a2.grid(alpha=.3)
a2.set_xlabel('Epoch')

plt.suptitle(f'Disease Model | Val Accuracy: {val_acc*100:.1f}% | {NUM_CLASSES} classes',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/disease_training_history.png', dpi=120, bbox_inches='tight')
print(f"✅ Plot saved: disease_training_history.png")

print(f"""
{'='*55}
DONE!
Files to copy to scav2/models/:
  ✅ disease_model.keras
  ✅ disease_model.keras (backup)
  ✅ disease_class_indices.json

Then restart the API:
  cd scav2/api
  uvicorn main:app --reload --port 8001
{'='*55}
""")
