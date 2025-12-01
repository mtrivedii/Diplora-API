import json
import os

# 1. Create Dummy Labels
labels = ["Normal Sinus Rhythm", "Atrial Fibrillation", "Other"]
with open("ecg_labels.json", "w") as f:
    json.dump(labels, f)
print("✅ Created ecg_labels.json")

# 2. Create Dummy Thresholds
thresholds = [0.5, 0.5, 0.5]
with open("best_thresholds.json", "w") as f:
    json.dump(thresholds, f)
print("✅ Created best_thresholds.json")

# 3. Create a Dummy TFLite Model 
# This creates an empty file just so the code finds it and doesn't crash.
with open("ecgnet_with_preprocessing.tflite", "wb") as f:
    f.write(b"dummy_model_content") 
print("✅ Created dummy ecgnet_with_preprocessing.tflite")