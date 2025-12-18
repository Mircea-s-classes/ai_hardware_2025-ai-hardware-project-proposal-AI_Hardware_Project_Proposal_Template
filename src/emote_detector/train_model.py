"""
Pose Classifier Training Pipeline
Generates training metrics, charts, and documentation for presentation

Run this AFTER collecting your own data with data_collector.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)


class PoseModelTrainer:
    """Training pipeline with full documentation and visualization"""
    
    def __init__(self, data_dir="pose_data", output_dir="results"):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing collected pose data
            output_dir: Directory for results, charts, and reports
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        
        self.pose_labels = {
            0: "Laughing",
            1: "Yawning",
            2: "Crying",
            3: "Taunting"
        }
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = {}
    
    def load_data(self):
        """Load collected training data"""
        print("\n" + "="*60)
        print("📦 STEP 1: Loading Training Data")
        print("="*60)
        
        features_file = self.data_dir / "pose_features_latest.npy"
        labels_file = self.data_dir / "pose_labels_latest.npy"
        metadata_file = self.data_dir / "pose_metadata_latest.json"
        
        if not features_file.exists():
            print(f"\n❌ No training data found in {self.data_dir}/")
            print("\nPlease collect data first:")
            print("  1. Run: python data_collector.py")
            print("  2. Press 0-3 to select pose")
            print("  3. Press 'a' to start auto-collection")
            print("  4. Collect ~50-100 samples per pose")
            print("  5. Press 's' to save")
            print("  6. Then run this script again")
            return False
        
        self.X = np.load(features_file)
        self.y = np.load(labels_file)
        
        # Load metadata if exists
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"collection_date": "Unknown"}
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Total samples: {len(self.X)}")
        print(f"   Features per sample: {self.X.shape[1]}")
        print(f"   Number of classes: {len(np.unique(self.y))}")
        
        # Print class distribution
        print(f"\n📊 Class Distribution:")
        print("-" * 40)
        unique, counts = np.unique(self.y, return_counts=True)
        for label, count in zip(unique, counts):
            name = self.pose_labels.get(label, f"Class {label}")
            pct = count / len(self.y) * 100
            print(f"   {name:<15}: {count:>4} samples ({pct:>5.1f}%)")
        print("-" * 40)
        print(f"   {'Total':<15}: {len(self.y):>4} samples")
        
        return True
    
    def visualize_data_distribution(self):
        """Create chart showing data distribution"""
        print("\n📊 Creating data distribution chart...")
        
        unique, counts = np.unique(self.y, return_counts=True)
        names = [self.pose_labels.get(l, f"Class {l}") for l in unique]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        
        ax.set_xlabel('Pose Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Training Data Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "data_distribution.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {chart_path}")
    
    def split_data(self, test_size=0.2):
        """Split data into training and test sets"""
        print("\n" + "="*60)
        print("📦 STEP 2: Splitting Data")
        print("="*60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        print(f"\n✅ Data split complete!")
        print(f"   Training samples: {len(self.X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"   Test samples: {len(self.X_test)} ({test_size*100:.0f}%)")
    
    def train_model(self, n_estimators=100, max_depth=10):
        """Train Random Forest classifier"""
        print("\n" + "="*60)
        print("🚀 STEP 3: Training Model")
        print("="*60)
        
        print(f"\n📋 Model Configuration:")
        print(f"   Algorithm: Random Forest Classifier")
        print(f"   Number of trees: {n_estimators}")
        print(f"   Max depth: {max_depth}")
        print(f"   Random state: 42")
        
        # Train model
        print(f"\n⏳ Training...")
        start_time = datetime.now()
        
        self.model = RandomForestClassifier(
            n_estimators=10,  # Ultra-light for RPi inference
            max_depth=4,      # Reduced for faster inference
            random_state=42,
            n_jobs=1  # Use single core for better inference performance on RPi
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Training complete!")
        print(f"   Training time: {training_time:.3f} seconds")
        
        # Store training history
        self.history['n_estimators'] = n_estimators
        self.history['max_depth'] = max_depth
        self.history['training_time'] = training_time
        self.history['n_features'] = self.X_train.shape[1]
    
    def evaluate_model(self):
        """Evaluate model and generate metrics"""
        print("\n" + "="*60)
        print("📊 STEP 4: Evaluating Model")
        print("="*60)
        
        # Training accuracy
        train_pred = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_pred)
        
        # Test accuracy
        test_pred = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        
        print(f"\n📈 Model Performance:")
        print("-" * 40)
        print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   Cross-Val Mean:    {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        print(f"   Cross-Val Std:     {cv_scores.std():.4f}")
        print("-" * 40)
        
        # Store results
        self.history['train_accuracy'] = train_acc
        self.history['test_accuracy'] = test_acc
        self.history['cv_mean'] = cv_scores.mean()
        self.history['cv_std'] = cv_scores.std()
        self.y_pred = test_pred
        
        return test_acc
    
    def plot_confusion_matrix(self):
        """Create confusion matrix visualization"""
        print("\n📊 Creating confusion matrix...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        labels = [self.pose_labels.get(i, f"Class {i}") for i in range(len(cm))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title('Confusion Matrix (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "confusion_matrix.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {chart_path}")
    
    def plot_per_class_accuracy(self):
        """Create per-class accuracy chart"""
        print("\n📊 Creating per-class accuracy chart...")
        
        # Calculate per-class accuracy
        cm = confusion_matrix(self.y_test, self.y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        
        labels = [self.pose_labels.get(i, f"Class {i}") for i in range(len(per_class_acc))]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, per_class_acc, 
                     color=['#27ae60' if acc >= 80 else '#f39c12' if acc >= 60 else '#e74c3c' 
                            for acc in per_class_acc])
        
        ax.set_xlabel('Pose Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        # Target line
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target (80%)')
        ax.legend()
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "per_class_accuracy.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {chart_path}")
        
        # Store
        self.history['per_class_accuracy'] = {labels[i]: per_class_acc[i] 
                                               for i in range(len(labels))}
    
    def plot_feature_importance(self):
        """Visualize feature importance from Random Forest"""
        print("\n📊 Creating feature importance chart...")
        
        importances = self.model.feature_importances_
        
        # Feature names
        feature_names = [
            'Shoulder Width', 'Hip Width', 'Body Height',
            'Left Arm Angle', 'Right Arm Angle',
            'Left Hand Height', 'Right Hand Height',
            'Left Knee Angle', 'Right Knee Angle',
            'Torso Angle', 'Left Arm Ext', 'Right Arm Ext',
            'Left Leg Angle', 'Right Leg Angle',
            'Body Symmetry', 'Left Hand Offset', 'Right Hand Offset',
            'Vertical Align'
        ][:len(importances)]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(sorted_importances)), sorted_importances, color='#3498db')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "feature_importance.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {chart_path}")
        
        # Store top features
        self.history['top_features'] = list(zip(sorted_names[:5], 
                                                sorted_importances[:5].tolist()))
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        print("\n📊 Generating classification report...")
        
        # Only use labels that exist in the data
        unique_labels = sorted(np.unique(np.concatenate([self.y_test, self.y_pred])))
        labels = [self.pose_labels.get(i, f"Class {i}") for i in unique_labels]
        
        report = classification_report(self.y_test, self.y_pred,
                                       labels=unique_labels,
                                       target_names=labels,
                                       digits=4)
        
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        print(report)
        print("="*60)
        
        # Save to file
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("Pose Classification Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(report)
        
        print(f"   ✅ Saved: {report_path}")
    
    def save_model(self, model_path="pose_classifier_model.pkl"):
        """Save trained model"""
        print("\n" + "="*60)
        print("💾 STEP 5: Saving Model")
        print("="*60)
        
        model_path = Path(model_path)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'pose_labels': self.pose_labels,
                'n_features': self.X.shape[1],
                'trained_date': datetime.now().isoformat()
            }, f)
        
        model_size = model_path.stat().st_size / 1024
        print(f"\n✅ Model saved!")
        print(f"   Path: {model_path}")
        print(f"   Size: {model_size:.2f} KB")
        
        self.history['model_size_kb'] = model_size
    
    def save_training_summary(self):
        """Save complete training summary"""
        print("\n📄 Saving training summary...")
        
        # Add metadata
        self.history['total_samples'] = len(self.X)
        self.history['training_samples'] = len(self.X_train)
        self.history['test_samples'] = len(self.X_test)
        self.history['num_classes'] = len(self.pose_labels)
        self.history['pose_labels'] = self.pose_labels
        self.history['timestamp'] = datetime.now().isoformat()
        
        # Save as JSON
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        
        # Save as readable text
        text_path = self.output_dir / "training_results.txt"
        with open(text_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("POSE CLASSIFIER TRAINING RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("📊 Dataset Information:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Total samples: {self.history['total_samples']}\n")
            f.write(f"  Training samples: {self.history['training_samples']}\n")
            f.write(f"  Test samples: {self.history['test_samples']}\n")
            f.write(f"  Number of features: {self.history['n_features']}\n")
            f.write(f"  Number of classes: {self.history['num_classes']}\n\n")
            
            f.write("📋 Model Configuration:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Algorithm: Random Forest\n")
            f.write(f"  Number of trees: {self.history['n_estimators']}\n")
            f.write(f"  Max depth: {self.history['max_depth']}\n")
            f.write(f"  Training time: {self.history['training_time']:.3f} seconds\n\n")
            
            f.write("📈 Performance Metrics:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Training Accuracy: {self.history['train_accuracy']*100:.2f}%\n")
            f.write(f"  Test Accuracy: {self.history['test_accuracy']*100:.2f}%\n")
            f.write(f"  Cross-Val Mean: {self.history['cv_mean']*100:.2f}%\n")
            f.write(f"  Cross-Val Std: {self.history['cv_std']*100:.2f}%\n\n")
            
            f.write("🎯 Per-Class Accuracy:\n")
            f.write("-"*40 + "\n")
            for pose, acc in self.history['per_class_accuracy'].items():
                f.write(f"  {pose}: {acc:.2f}%\n")
            
            f.write("\n🔑 Top Important Features:\n")
            f.write("-"*40 + "\n")
            for i, (name, imp) in enumerate(self.history['top_features'], 1):
                f.write(f"  {i}. {name}: {imp:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"   ✅ Saved: {summary_path}")
        print(f"   ✅ Saved: {text_path}")
    
    def print_final_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        
        print(f"\n📊 Results Summary:")
        print(f"   Training Accuracy: {self.history['train_accuracy']*100:.2f}%")
        print(f"   Test Accuracy: {self.history['test_accuracy']*100:.2f}%")
        print(f"   Training Time: {self.history['training_time']:.3f} seconds")
        print(f"   Model Size: {self.history['model_size_kb']:.2f} KB")
        
        print(f"\n📁 Generated Files:")
        print(f"   Results: {self.output_dir}/")
        print(f"   - training_results.txt")
        print(f"   - training_summary.json")
        print(f"   - classification_report.txt")
        print(f"   Charts: {self.output_dir}/charts/")
        print(f"   - data_distribution.png")
        print(f"   - confusion_matrix.png")
        print(f"   - per_class_accuracy.png")
        print(f"   - feature_importance.png")
        
        print("\n✅ Ready to run the detector:")
        print("   python main.py")
        print("="*60)


def main():
    """Run full training pipeline"""
    print("\n" + "="*60)
    print("🎓 POSE CLASSIFIER TRAINING PIPELINE")
    print("For AI Hardware Class Presentation")
    print("="*60)
    
    trainer = PoseModelTrainer()
    
    # Step 1: Load data
    if not trainer.load_data():
        return
    
    # Create data distribution chart
    trainer.visualize_data_distribution()
    
    # Step 2: Split data
    trainer.split_data(test_size=0.2)
    
    # Step 3: Train model
    trainer.train_model(n_estimators=100, max_depth=10)
    
    # Step 4: Evaluate
    trainer.evaluate_model()
    
    # Generate all visualizations
    trainer.plot_confusion_matrix()
    trainer.plot_per_class_accuracy()
    trainer.plot_feature_importance()
    trainer.generate_classification_report()
    
    # Step 5: Save model
    trainer.save_model()
    
    # Save summary
    trainer.save_training_summary()
    
    # Final summary
    trainer.print_final_summary()


if __name__ == "__main__":
    main()

