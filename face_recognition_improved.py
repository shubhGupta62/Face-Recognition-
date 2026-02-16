import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_FACES_DIR = r"c:\Users\gshub\OneDrive\Desktop\project 2\dataset\faces"
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👤 Face Recognition System - In-House Dataset")
        self.root.geometry("1100x850")
        self.root.configure(bg='#f0f0f0')
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        
        # Initialize ORB detector for better feature matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Prepare dataset
        self.face_database = {}
        self.load_dataset()
        
        self.setup_ui()
        
    def load_dataset(self):
        """Load all faces from in-house dataset"""
        print("=" * 90)
        print("LOADING IN-HOUSE FACE DATABASE FROM DATASET/FACES")
        print("=" * 90)
        
        if not os.path.exists(DATASET_FACES_DIR):
            print(f"❌ Error: Dataset directory not found at {DATASET_FACES_DIR}")
            return
        
        person_folders = sorted([d for d in os.listdir(DATASET_FACES_DIR) 
                                if os.path.isdir(os.path.join(DATASET_FACES_DIR, d))])
        
        print(f"\nFound {len(person_folders)} person categories:")
        print(f"  {', '.join(person_folders)}\n")
        
        total_faces = 0
        
        for person_name in person_folders:
            person_path = os.path.join(DATASET_FACES_DIR, person_name)
            image_files = sorted([f for f in os.listdir(person_path) 
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            face_features = []
            valid_count = 0
            
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                    
                    if len(faces) > 0:
                        # Extract and process face
                        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                        
                        # Add margin for better recognition
                        margin = int(max(w, h) * 0.2)
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(img.shape[1], x + w + margin)
                        y2 = min(img.shape[0], y + h + margin)
                        
                        face_roi = img[y1:y2, x1:x2]
                        
                        # Normalize face
                        face_roi = cv2.resize(face_roi, (160, 160))
                        
                        # Extract features
                        features = self.extract_features(face_roi)
                        if features is not None:
                            face_features.append(features)
                            valid_count += 1
                            total_faces += 1
                        
                except Exception as e:
                    pass
            
            if face_features:
                # Store aggregated features
                avg_features = np.mean(face_features, axis=0)
                self.face_database[person_name] = {
                    'features': avg_features,
                    'num_samples': valid_count,
                    'path': person_path
                }
                print(f"  ✓ {person_name:15} | Faces: {valid_count:3} | Features Extracted")
        
        print(f"\n{'=' * 90}")
        print(f"✓ DATABASE READY")
        print(f"  Total Faces Loaded:  {total_faces}")
        print(f"  Total Persons:       {len(self.face_database)}")
        print(f"  Dataset Location:    {DATASET_FACES_DIR}")
        print(f"{'=' * 90}\n")
    
    def extract_features(self, face_image):
        """Extract more discriminative face features"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # 1. SIFT-like features (ORB with more detail)
            orb = cv2.ORB_create(nfeatures=2000)  # More keypoints for better discrimination
            kp, des = orb.detectAndCompute(gray, None)
            
            if des is None or len(kp) < 5:
                return self.extract_fallback_features(face_image)
            
            # Use actual descriptor values (not just mean)
            des_float = des.astype(np.float32) / 255.0
            des_vector = des_float.flatten()[:1000]  # Take first 1000 values
            
            # Pad if needed
            if len(des_vector) < 1000:
                des_vector = np.pad(des_vector, (0, 1000 - len(des_vector)))
            
            # 2. LBP texture features - more discriminative
            lbp = self.compute_lbp(gray)
            
            # 3. Face region analysis - eyes, nose, mouth areas
            h, w = gray.shape
            regions = []
            
            # Top region (eyes area)
            top_region = gray[:h//3, :].flatten()
            regions.append(np.histogram(top_region, bins=32)[0])
            
            # Middle region (nose area)
            mid_region = gray[h//3:2*h//3, :].flatten()
            regions.append(np.histogram(mid_region, bins=32)[0])
            
            # Bottom region (mouth area)
            bot_region = gray[2*h//3:, :].flatten()
            regions.append(np.histogram(bot_region, bins=32)[0])
            
            regions = np.concatenate(regions)
            
            # 4. Detailed edge features at multiple scales
            edges_1 = cv2.Canny(gray, 30, 90)
            edges_2 = cv2.Canny(gray, 50, 150)
            edges_3 = cv2.Canny(gray, 100, 200)
            
            edge_features = np.concatenate([
                edges_1.flatten()[:500],
                edges_2.flatten()[:500],
                edges_3.flatten()[:500]
            ])
            
            # 5. Color space features (multiple color spaces)
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
            
            color_features = np.concatenate([h_hist, s_hist, v_hist])
            
            # 6. Gradient magnitude and direction
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)
            
            mag_hist = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
            dir_hist = np.histogram(direction.flatten(), bins=32)[0]
            
            gradient_features = np.concatenate([mag_hist, dir_hist])
            
            # Combine all features
            all_features = np.concatenate([
                des_vector,
                lbp[:256],
                regions,
                edge_features,
                color_features,
                gradient_features
            ])
            
            # Normalize carefully (avoid making dissimilar faces too similar)
            norm = np.linalg.norm(all_features)
            if norm > 0:
                all_features = all_features / norm
            
            return all_features
            
        except Exception as e:
            return self.extract_fallback_features(face_image)
    
    def compute_lbp(self, gray_image):
        """Compute Local Binary Pattern"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    lbp_val = 0
                    
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1],
                        gray_image[i+1, j], gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor > center:
                            lbp_val |= (1 << k)
                    
                    lbp[i-1, j-1] = lbp_val
            
            hist = np.histogram(lbp.flatten(), bins=256, range=(0, 256))[0]
            return hist / (hist.sum() + 1e-6)
        except:
            return np.zeros(256)
    
    def extract_fallback_features(self, face_image):
        """Fallback feature extraction"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Simple histogram-based features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # HSV
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        
        # Edge
        edges = cv2.Canny(gray, 50, 150)
        edge_features = edges.flatten()[:1000]
        if len(edge_features) < 1000:
            edge_features = np.pad(edge_features, (0, 1000 - len(edge_features)))
        
        features = np.concatenate([hist, h_hist, edge_features])
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def setup_ui(self):
        """Setup user interface"""
        # Title
        title = tk.Label(self.root, text="👤 FACE RECOGNITION SYSTEM",
                        font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#1a1a1a')
        title.pack(pady=15)
        
        subtitle = tk.Label(self.root, text="In-House Dataset | 9 Persons × 358+ Face Images",
                           font=("Arial", 12), bg='#f0f0f0', fg='#555555')
        subtitle.pack()
        
        # Button frame
        btn_frame = tk.Frame(self.root, bg='#f0f0f0')
        btn_frame.pack(pady=15)
        
        upload_btn = tk.Button(btn_frame, text="📁 UPLOAD IMAGE",
                              command=self.upload_image,
                              font=("Arial", 13, "bold"),
                              bg='#27ae60', fg='white', padx=25, pady=12,
                              cursor="hand2", relief=tk.RAISED, bd=2,
                              activebackground='#229954')
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(btn_frame, text="🗑️ CLEAR",
                             command=self.clear,
                             font=("Arial", 13, "bold"),
                             bg='#e74c3c', fg='white', padx=25, pady=12,
                             cursor="hand2", relief=tk.RAISED, bd=2,
                             activebackground='#c0392b')
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Main content
        content = tk.Frame(self.root, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Image panel
        img_panel = tk.LabelFrame(content, text="🖼️  INPUT IMAGE",
                                 font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#1a1a1a',
                                 relief=tk.RIDGE, bd=2)
        img_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(img_panel, bg='#ecf0f1', relief=tk.SUNKEN, bd=2)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results panel
        res_panel = tk.LabelFrame(content, text="🎯 RECOGNITION RESULTS",
                                 font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#1a1a1a',
                                 relief=tk.RIDGE, bd=2)
        res_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.results_text = tk.Text(res_panel, font=("Courier", 10),
                                   bg='#ecf0f1', fg='#1a1a1a', relief=tk.FLAT,
                                   wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status = tk.Label(self.root, text="✓ Ready | Waiting for image",
                              font=("Arial", 11), bg='#27ae60', fg='white',
                              relief=tk.SUNKEN, anchor='w', pady=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """Upload and process image"""
        path = filedialog.askopenfilename(
            title="Select face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")]
        )
        
        if path:
            thread = threading.Thread(target=self.process_image, args=(path,))
            thread.daemon = True
            thread.start()
    
    def process_image(self, path):
        """Process uploaded image"""
        try:
            self.status.config(text="⏳ Processing...", bg='#f39c12')
            self.root.update()
            
            # Read image
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Cannot read image")
                self.status.config(text="✗ Error", bg='#e74c3c')
                return
            
            # Display image
            self.show_image(img)
            
            # Detect face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            if len(faces) == 0:
                result = "❌ NO FACE DETECTED\n\n"
                result += "Please upload a clear image with a frontal face view."
                self.show_results(result)
                self.status.config(text="✗ No face detected", bg='#e74c3c')
                return
            
            # Extract face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            
            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            
            # Extract features
            features = self.extract_features(face)
            if features is None:
                self.show_results("❌ Could not extract face features")
                self.status.config(text="✗ Feature extraction failed", bg='#e74c3c')
                return
            
            # Compare with database
            result = self.recognize(features)
            self.show_results(result)
            self.status.config(text="✓ Complete", bg='#27ae60')
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="✗ Error", bg='#e74c3c')
    
    def recognize(self, test_features):
        """Match face with database using better metrics"""
        scores = {}
        
        for person, data in self.face_database.items():
            db_features = data['features']
            
            # Calculate multiple similarity metrics
            cosine_sim = cosine_similarity([test_features], [db_features])[0][0]
            euclidean_dist = np.linalg.norm(test_features - db_features)
            
            # Combine metrics for better discrimination
            # Lower euclidean distance = better match
            euclidean_score = 1.0 / (1.0 + euclidean_dist)
            
            # Weighted combination
            combined_score = (cosine_sim * 0.6 + euclidean_score * 0.4)
            
            scores[person] = {
                'cosine': cosine_sim,
                'euclidean': euclidean_score,
                'combined': combined_score
            }
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['combined'], reverse=True)
        
        # Build result
        result = "╔" + "═" * 66 + "╗\n"
        result += "║" + " FACE RECOGNITION RESULTS ".center(66) + "║\n"
        result += "╚" + "═" * 66 + "╝\n\n"
        
        top = sorted_scores[0]
        top_person = top[0]
        top_score = top[1]['combined']
        cosine = top[1]['cosine']
        
        # Improved confidence calculation
        second_score = sorted_scores[1][1]['combined'] if len(sorted_scores) > 1 else 0
        margin = top_score - second_score
        
        if margin > 0.3:
            status = "🟢 VERY HIGH CONFIDENCE"
        elif margin > 0.15:
            status = "🟢 HIGH CONFIDENCE"  
        elif margin > 0.08:
            status = "🟡 MEDIUM CONFIDENCE"
        else:
            status = "🔴 LOW CONFIDENCE - WEAK MATCH"
        
        result += f"👤 PERSON IDENTIFIED: {top_person.upper()}\n"
        result += f"   Similarity Score: {top_score*100:.2f}%\n"
        result += f"   Confidence Level: {status}\n"
        result += f"   Database Sample Size: {self.face_database[top_person]['num_samples']} faces\n"
        result += f"   Match Margin (vs 2nd): {margin*100:.2f}%\n\n"
        
        result += "┌" + "─" * 66 + "┐\n"
        result += "│ TOP 5 MATCHES\n"
        result += "├" + "─" * 66 + "┤\n"
        
        for i, (person, score_dict) in enumerate(sorted_scores[:5], 1):
            score = score_dict['combined']
            bar_len = int(score * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            result += f"│ {i}. {person:15} {bar} {score*100:6.2f}%\n"
        
        result += "└" + "─" * 66 + "┘"
        
        return result
    
    def show_image(self, cv_img):
        """Display image"""
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if w > h:
            new_w = 400
            new_h = int(400 * h / w)
        else:
            new_h = 400
            new_w = int(400 * w / h)
        
        img = cv2.resize(img, (new_w, new_h))
        pil_img = Image.fromarray(img)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img
    
    def show_results(self, text):
        """Display results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        self.results_text.config(state=tk.DISABLED)
    
    def clear(self):
        """Clear all"""
        self.image_label.config(image='')
        self.image_label.image = None
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.status.config(text="✓ Ready", bg='#27ae60')

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
