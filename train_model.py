import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class ASLModelTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        self.model = None
        self.classes = []
        
    def extract_landmarks(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            return None
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def load_dataset(self, data_path):
        X = []
        y = []
        self.classes = []
        
        print("Loading dataset from:", data_path)
        
        if not os.path.exists(data_path):
            print(f"❌ ERROR: Dataset path '{data_path}' does not exist!")
            return X, y
        
        class_folders = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
        
        print(f"Found {len(class_folders)} class folders: {class_folders}")
        
        for class_name in class_folders:
            class_path = os.path.join(data_path, class_name)
            self.classes.append(class_name)
            print(f"📁 Processing class: {class_name}")
            
            image_count = 0
            valid_images = 0
            
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"   ⚠️  No images found in {class_name} folder")
                continue
                
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"   ❌ Could not read image: {image_file}")
                    continue
                
                image_count += 1
                    
                landmarks = self.extract_landmarks(image)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(class_name)
                    valid_images += 1
                else:
                    print(f"   ⚠️  No hand detected in: {image_file}")
            
            print(f"   ✅ Processed {image_count} images, {valid_images} with valid hand landmarks")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total classes: {len(self.classes)}")
        print(f"   Total samples: {len(X)}")
        return np.array(X), np.array(y)
    
    def train(self, data_path="asl_dataset", model_save_path="model/asl_model.pkl"):
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        print("🚀 Starting ASL Model Training...")
        print(f"📁 Dataset path: {data_path}")
        print(f"💾 Model will be saved to: {model_save_path}")
        
        X, y = self.load_dataset(data_path)
        
        if len(X) == 0:
            print("❌ No training data found! Please check:")
            print("   1. Dataset path is correct")
            print("   2. Images are in proper format (jpg, png, jpeg)")
            print("   3. Hands are visible in images")
            return False
        
        print(f"\n🎯 Training on {len(X)} samples across {len(self.classes)} classes")
        print(f"📝 Classes: {sorted(self.classes)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training Complete!")
        print(f"📈 Model accuracy: {accuracy:.4f} ({accuracy:.2%})")
        
        print(f"\n🔍 Sample predictions:")
        for i in range(min(5, len(X_test))):
            print(f"   True: {y_test[i]}, Predicted: {y_pred[i]}")
        
        model_data = {
            'model': self.model,
            'classes': self.classes,
            'accuracy': accuracy
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to: {model_save_path}")
        print(f"🎯 Classes learned: {sorted(self.classes)}")
        
        return True

def main():
    dataset_path = "asl_dataset"
    
    trainer = ASLModelTrainer()
    success = trainer.train(dataset_path)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("👉 You can now run: streamlit run app.py")
    else:
        print("\n❌ Training failed! Please check the errors above.")

if __name__ == "__main__":
    main()
