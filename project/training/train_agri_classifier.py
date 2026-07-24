import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def train_classifier():
    print("Training generic agricultural vs non-agricultural classifier on v_2 dataset...")
    
    workspace_dir = r"D:\PC\resources"
    v2_dir = os.path.join(workspace_dir, "v_2")
    classes = ["agri", "barrenland", "grassland", "urban"]
    
    # We will sample 500 images from each class to keep it extremely fast
    num_samples = 500
    
    data = []
    labels = []
    
    for cls in classes:
        cls_dir = os.path.join(v2_dir, cls)
        s1_dir = os.path.join(cls_dir, "s1")
        s2_dir = os.path.join(cls_dir, "s2")
        
        s1_files = sorted(os.listdir(s1_dir))[:num_samples]
        s2_files = sorted(os.listdir(s2_dir))[:num_samples]
        
        # Agri is class 1, barrenland/grassland/urban are class 0 (non-agri)
        label = 1 if cls == "agri" else 0
        
        for f1, f2 in zip(s1_files, s2_files):
            p1 = os.path.join(s1_dir, f1)
            p2 = os.path.join(s2_dir, f2)
            
            img_s1 = cv2.imread(p1)
            img_s2 = cv2.imread(p2)
            
            if img_s1 is None or img_s2 is None:
                continue
                
            # Extract simple features: mean and standard deviation of each channel
            s1_mean = img_s1.mean(axis=(0,1))
            s1_std = img_s1.std(axis=(0,1))
            s2_mean = img_s2.mean(axis=(0,1))
            s2_std = img_s2.std(axis=(0,1))
            
            feat = np.concatenate([s1_mean, s1_std, s2_mean, s2_std])
            data.append(feat)
            labels.append(label)
            
    X = np.array(data)
    y = np.array(labels)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    
    val_preds = clf.predict(X_val)
    report = classification_report(y_val, val_preds, target_names=["Non-Agricultural", "Agricultural"])
    print("\nClassifier Performance Report:")
    print(report)
    
    # Save the model
    models_dir = os.path.join(workspace_dir, "project", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "generic_agri_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Generic classifier saved to: {model_path}")
    
    # Save classification report as a text file for the report
    report_path = os.path.join(workspace_dir, "project", "outputs", "agri_classification_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")

if __name__ == "__main__":
    train_classifier()
