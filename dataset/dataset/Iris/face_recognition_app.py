import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk
import threading

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_FACES_DIR = r"c:\Users\gshub\OneDrive\Desktop\project 2\dataset\faces"

# Recognition threshold: lower = stricter matching
# Euclidean distance in embedding space; typical range 0.0 - 2.0
RECOGNITION_THRESHOLD = 1.0

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')

        # ---------------------------------------------------
        # Deep learning models (facenet-pytorch)
        # ---------------------------------------------------
        # MTCNN: Multi-task Cascaded Convolutional Networks
        #   - Much more accurate than Haar cascade
        #   - Handles face alignment automatically
        #   - Returns cropped, aligned 160x160 face tensors
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=True,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            device=DEVICE
        )

        # InceptionResnetV1: Deep face embedding network
        #   - Pretrained on VGGFace2 dataset (3.3M images, 9131 subjects)
        #   - Produces 512-dimensional face embeddings
        #   - State-of-the-art accuracy for face verification
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

        # Face database: {person_name: [list of 512-d embedding tensors]}
        self.face_database = {}
        self.load_dataset()

        self.setup_ui()

    # ========================================================
    # DATASET LOADING
    # ========================================================
    def load_dataset(self):
        """Load all faces from in-house dataset using deep learning."""
        print("=" * 70)
        print("LOADING FACE DATABASE (Deep Learning Embeddings)")
        print("=" * 70)

        if not os.path.exists(DATASET_FACES_DIR):
            print(f"ERROR: Dataset directory not found: {DATASET_FACES_DIR}")
            return

        person_folders = sorted([
            d for d in os.listdir(DATASET_FACES_DIR)
            if os.path.isdir(os.path.join(DATASET_FACES_DIR, d))
        ])

        total_faces = 0

        for person_name in person_folders:
            person_path = os.path.join(DATASET_FACES_DIR, person_name)
            image_files = [
                f for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
            ]

            embeddings = []

            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                try:
                    # Load image as RGB PIL Image (required by MTCNN)
                    pil_img = Image.open(img_path).convert('RGB')

                    # Detect and align faces using MTCNN
                    # Returns tensor of shape (n_faces, 3, 160, 160)
                    faces, probs = self.mtcnn(pil_img, return_prob=True)

                    if faces is not None and len(faces) > 0:
                        # Pick the face with highest detection confidence
                        best_idx = int(np.argmax(probs))
                        face_tensor = faces[best_idx].unsqueeze(0).to(DEVICE)

                        # Generate 512-d embedding using InceptionResnetV1
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor)

                        embeddings.append(embedding.cpu().squeeze())
                        total_faces += 1

                except Exception as e:
                    # Skip corrupt / unreadable images
                    continue

            if embeddings:
                self.face_database[person_name] = embeddings
                print(f"  ✓ {person_name}: {len(embeddings)} embeddings")

        print(f"\nTotal face embeddings: {total_faces}")
        print(f"Unique persons: {len(self.face_database)}")
        print("=" * 70 + "\n")

    # ========================================================
    # FACE RECOGNITION
    # ========================================================
    def recognize_face(self, image_path):
        """Recognize face in an uploaded image."""
        try:
            self.status_label.config(text="Processing image...")
            self.root.update()

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Could not read image file")
                return

            # Display the uploaded image in the UI
            self.display_image(image_path, img)

            # Convert to RGB PIL Image for MTCNN
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Detect faces with MTCNN
            faces, probs = self.mtcnn(pil_img, return_prob=True)

            if faces is None or len(faces) == 0:
                results = "❌ NO FACES DETECTED\n\nPlease upload an image with a clear face."
                self.display_results(results)
                self.status_label.config(text="No faces found")
                return

            # Use the face with highest detection probability
            best_idx = int(np.argmax(probs))
            face_tensor = faces[best_idx].unsqueeze(0).to(DEVICE)
            detection_conf = float(probs[best_idx])

            # Generate embedding for test face
            with torch.no_grad():
                test_embedding = self.resnet(face_tensor).cpu().squeeze()

            # Compare against database
            results = self.compare_with_database(test_embedding, detection_conf)
            self.display_results(results)
            self.status_label.config(text="Recognition completed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
            self.status_label.config(text="Error during recognition")

    def compare_with_database(self, test_embedding, detection_conf):
        """
        Compare test embedding against ALL stored embeddings.

        Uses Euclidean distance (L2 norm) — the standard metric for
        FaceNet-style embeddings. Lower distance = more similar.
        """
        if not self.face_database:
            return "❌ DATABASE EMPTY\n\nNo faces loaded from dataset."

        # For each person, find the MINIMUM distance across all their embeddings
        # This is more robust than comparing against an average embedding
        person_scores = {}

        for person_name, embeddings_list in self.face_database.items():
            distances = []
            for db_emb in embeddings_list:
                dist = torch.dist(test_embedding, db_emb, p=2).item()
                distances.append(dist)

            # Best (smallest) distance for this person
            min_dist = min(distances)
            # Also compute average of top-3 closest for robustness
            top3 = sorted(distances)[:3]
            avg_top3 = sum(top3) / len(top3)

            person_scores[person_name] = {
                'min_dist': min_dist,
                'avg_top3': avg_top3,
                'n_samples': len(embeddings_list),
            }

        # Sort by minimum distance (ascending = best match first)
        sorted_scores = sorted(
            person_scores.items(),
            key=lambda x: x[1]['min_dist']
        )

        # ---- Build results string ----
        results = "=" * 60 + "\n"
        results += "FACE RECOGNITION RESULTS\n"
        results += "=" * 60 + "\n\n"

        top_match_name, top_match_data = sorted_scores[0]
        best_dist = top_match_data['min_dist']

        # Convert distance to a confidence percentage
        # distance=0 → 100%, distance=THRESHOLD → 0%
        confidence = max(0, (1 - best_dist / RECOGNITION_THRESHOLD)) * 100

        if best_dist < RECOGNITION_THRESHOLD:
            results += f"🎯 IDENTIFIED: {top_match_name}\n"
            results += f"   Confidence : {confidence:.1f}%\n"
            results += f"   Distance   : {best_dist:.4f}\n"
            results += f"   Samples    : {top_match_data['n_samples']}\n"
            results += f"   Face Detect: {detection_conf:.2%}\n\n"
        else:
            results += "⚠️  UNKNOWN PERSON\n"
            results += f"   Best distance ({best_dist:.4f}) exceeds "
            results += f"threshold ({RECOGNITION_THRESHOLD})\n"
            results += f"   Closest match was: {top_match_name}\n\n"

        results += "─" * 60 + "\n"
        results += "ALL MATCHES (sorted by distance, lower = better):\n"
        results += "─" * 60 + "\n"

        for i, (person, data) in enumerate(sorted_scores, 1):
            dist = data['min_dist']
            conf = max(0, (1 - dist / RECOGNITION_THRESHOLD)) * 100
            bar_len = int(max(0, conf) * 0.3)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ✓" if dist < RECOGNITION_THRESHOLD else ""
            results += f"{i:2d}. {person:12s} {bar} dist={dist:.3f}  {conf:5.1f}%{marker}\n"

        return results

    # ========================================================
    # UI
    # ========================================================
    def setup_ui(self):
        """Setup UI components."""
        title_label = tk.Label(
            self.root, text="FACE RECOGNITION SYSTEM (Deep Learning)",
            font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#2c3e50'
        )
        title_label.pack(pady=10)

        # Info bar
        db_info = f"Database: {len(self.face_database)} persons | "
        db_info += f"Model: InceptionResnetV1 (VGGFace2) | "
        db_info += f"Detector: MTCNN"
        info_label = tk.Label(
            self.root, text=db_info,
            font=("Arial", 9), bg='#f0f0f0', fg='#7f8c8d'
        )
        info_label.pack()

        # Buttons
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)

        upload_btn = tk.Button(
            button_frame, text="📁 Upload Test Image",
            command=self.upload_image,
            font=("Arial", 12, "bold"),
            bg='#3498db', fg='white', padx=15, pady=10,
            cursor="hand2"
        )
        upload_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            button_frame, text="🗑️ Clear",
            command=self.clear_results,
            font=("Arial", 12, "bold"),
            bg='#e74c3c', fg='white', padx=15, pady=10,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Image display
        image_frame = tk.LabelFrame(
            self.root, text="Test Image",
            font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50'
        )
        image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame, bg='#ecf0f1', height=15)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Results display
        results_frame = tk.LabelFrame(
            self.root, text="Recognition Results",
            font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50'
        )
        results_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(
            results_frame, height=10, font=("Courier", 10),
            bg='#ecf0f1', fg='#2c3e50', relief=tk.FLAT
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status bar
        self.status_label = tk.Label(
            self.root, text="Ready — Deep Learning Model Loaded",
            font=("Arial", 10), bg='#2c3e50', fg='white',
            relief=tk.SUNKEN, anchor='w'
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        """Handle image upload."""
        file_path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            thread = threading.Thread(
                target=self.recognize_face, args=(file_path,)
            )
            thread.start()

    def display_image(self, image_path, cv_image):
        """Display image in UI."""
        display_img = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2RGB)
        h, w = display_img.shape[:2]
        if w > h:
            new_w, new_h = 400, int(400 * h / w)
        else:
            new_h, new_w = 400, int(400 * w / h)
        display_img = cv2.resize(display_img, (new_w, new_h))
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

    def display_results(self, results):
        """Display results in text widget."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.results_text.config(state=tk.DISABLED)

    def clear_results(self):
        """Clear all results."""
        self.image_label.config(image='')
        self.image_label.image = None
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FACE RECOGNITION — Deep Learning Pipeline")
    print("  Detector : MTCNN")
    print("  Encoder  : InceptionResnetV1 (VGGFace2)")
    print("  Dataset  :", DATASET_FACES_DIR)
    print("=" * 70 + "\n")
    main()
