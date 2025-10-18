import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

class ASLDetector:
    def __init__(self, model_path="model/asl_model.pkl"):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.model = None
        self.classes = []
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.classes = model_data['classes']
                print(f"✅ Model loaded successfully!")
                print(f"📚 Classes: {len(self.classes)} signs")
                print(f"🎯 Supported: {sorted(self.classes)}")
            else:
                print("❌ Model file not found! Please train the model first.")
                print("💡 Run: python train_model.py")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def extract_landmarks(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    return np.array(landmarks), results.multi_hand_landmarks
            return None, None
        except Exception as e:
            print(f"Error in extract_landmarks: {e}")
            return None, None
    
    def predict(self, image):
        if self.model is None:
            return "Model not loaded", 0, None
        
        landmarks, hand_landmarks = self.extract_landmarks(image)
        
        if landmarks is not None:
            try:
                prediction = self.model.predict([landmarks])[0]
                probabilities = self.model.predict_proba([landmarks])[0]
                confidence = np.max(probabilities)
                
                return prediction, confidence, hand_landmarks
            except Exception as e:
                print(f"Prediction error: {e}")
                return "Prediction error", 0, hand_landmarks
        else:
            return "No hand detected", 0, None
    
    def draw_landmarks(self, image, hand_landmarks):
        try:
            annotated_image = image.copy()
            for landmarks in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            return annotated_image
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
            return image
