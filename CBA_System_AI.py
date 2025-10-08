import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import getpass
from pathlib import Path
from datetime import datetime

# Prepare directory structure - using temp directory for Streamlit cloud
base_dir = Path(tempfile.gettempdir()) / "results"
images_dir = base_dir / "images"
videos_dir = base_dir / "videos"

# Create directories if they don't exist
base_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(exist_ok=True)
videos_dir.mkdir(exist_ok=True)

# Loading all YOLOv8n Models
try:
    #Weapon Model Detects Guns and Knives
    gun_model = YOLO('./Models/detect/Guns & Knives/weights/best.pt')
    #Placard Model Detects Placards and Knives  
    placard_model = YOLO('./Models/detect/Placard-stick/weights/best.pt')  
    #Behaviour Model Detects if someone is Fighting or in Fighting Posture
    behavior_model = YOLO('./Models/detect/Fights/weights/best.pt') 
    st.success("All Three YOLOv8 Models Loaded Succesfully!")
except Exception as e:
    st.error(f"Error loading YOLO models: {e}")
    st.stop()

# Custom function to draw bounding boxes and labels from all models
def draw_combined_results(frame, gun_results, placard_results, behavior_results):
    combined_frame = frame.copy()
    
    # Define fixed colors for each class
    CLASS_COLORS = {
        "Guns": (0, 0, 255),       # Red
        "Knives": (0, 0, 255),      # Red
        "Placards": (0, 255, 0),    # Green
        "Sticks": (255, 0, 0),      # Blue
        "Fight": (128, 0, 128),    # Purple
    }
    
    # Define confidence thresholds for each model
    GUN_CONF_THRESHOLD = st.session_state.get('conf_threshold_gun', 0.5)  
    PLACARD_CONF_THRESHOLD = st.session_state.get('conf_threshold_placard', 0.55)  
    BEHAVIOR_CONF_THRESHOLD = st.session_state.get('conf_threshold_behavior', 0.6)  
    
    # Drawing Results of Fun model
    if gun_results and len(gun_results) > 0:
        gun_result = gun_results[0]
        if gun_result.boxes:
            for box in gun_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = gun_model.names[cls]
                print(f"[Gun Model] Detected {class_name} at x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.2f}")
                if conf < GUN_CONF_THRESHOLD:
                    continue  # Skip detections below threshold
                color = CLASS_COLORS.get(class_name, (0, 0, 255))  # Default: Red
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(combined_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Drawing results from Placard model
    if placard_results and len(placard_results) > 0:
        placard_result = placard_results[0]
        if placard_result.boxes:
            for box in placard_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = placard_model.names[cls]
                print(f"[Placard Model] Detected {class_name} at x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.2f}")
                if conf < PLACARD_CONF_THRESHOLD:
                    continue  # Skip detections below threshold
                color = CLASS_COLORS.get(class_name, (0, 255, 0))  # Default: Green
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(combined_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Drawing results from Behavior model
    if behavior_results and len(behavior_results) > 0:
        behavior_result = behavior_results[0]
        if behavior_result.boxes:
            for box in behavior_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = "Fight"  # Force label to be "Fight"
                print(f"[Behavior Model] Detected {class_name} at x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf:.2f}")
                if conf < BEHAVIOR_CONF_THRESHOLD:
                    continue  # Skip detections below threshold
                color = CLASS_COLORS.get(class_name, (128, 0, 128))  # Default: Purple
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(combined_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return combined_frame

# Function to process and annotate an uploaded image
def detect_image(image_file):
    try:
        # Read image from uploaded file
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Not able to detect image. Please check the file format.")
            return None, None

        # Run all three models
        gun_results = gun_model(img, verbose=False)
        placard_results = placard_model(img, verbose=False)
        behavior_results = behavior_model(img, verbose=False)

        # Draw combined results
        annotated_img = draw_combined_results(img, gun_results, placard_results, behavior_results)

        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"detected_{timestamp}.jpg"
        output_path = images_dir / output_filename
        cv2.imwrite(str(output_path), annotated_img)

        st.success(f"Image processed, Output saved {output_path}")
        return annotated_img, str(output_path.resolve())
    except Exception as e:
        st.error(f"Error Processing Image {e}")
        return None, None

def detect_video(video_file):
    try:
        st.write("Video Processing started...")
        
        # Create temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_data = video_file.read()
        tfile.write(video_data)
        tfile.close()
        
        if not os.path.exists(tfile.name) or os.path.getsize(tfile.name) == 0:
            st.error("Error: Video File is Empty or Invalid...")
            return None
        
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error(f"Error: Video {video_file.name} didn't open")
            os.unlink(tfile.name)
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.total_frames = total_frames
        st.write(f"Total frames in Video: {total_frames}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"detected_{timestamp}.mp4"
        output_path = videos_dir / output_filename

        # Use MP4V codec which is more widely supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            st.error("Error: Video writer not initialized. Trying alternative approach...")
            # Try alternative approach
            output_path = str(output_path) + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                st.error("Failed to initialize video writer with both methods")
                cap.release()
                os.unlink(tfile.name)
                return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_count = 0
        st.session_state.frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Run all three models on the frame
                gun_results = gun_model(frame, verbose=False)
                placard_results = placard_model(frame, verbose=False)
                behavior_results = behavior_model(frame, verbose=False)
                
                # Combine results from all three
                annotated_frame = draw_combined_results(frame, gun_results, placard_results, behavior_results)
                
                # Write annotated frame to output video
                out.write(annotated_frame)
                frame_count += 1
                st.session_state.frame_count = frame_count
                
                if frame_count % 10 == 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                
                if frame_count % 100 == 0:
                    st.write(f"Processed {frame_count} frames")
                    
            except Exception as e:
                st.write(f"Error processing frame {frame_count}: {e}")
                continue
        
        # Release resources
        cap.release()
        out.release()
        os.unlink(tfile.name)
        
        # Verify output file
        if os.path.exists(output_path):
            cap_test = cv2.VideoCapture(output_path)
            if cap_test.isOpened():
                st.write(f"Video validated: {cap_test.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
                cap_test.release()
            else:
                st.error("Output video is corrupt")
                return None
        else:
            st.error("Output video not created")
            return None
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Processed {frame_count}/{total_frames} frames")
        st.success(f"Video processing completed. Output saved: {output_path}")
        
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def detect_live_camera():
    try:
        # Check if running in browser (like Streamlit Sharing)
        is_browser = "streamlit" in st.__file__.lower()
        
        if is_browser:
            # For browser-based environments (like Streamlit Sharing)
            st.warning("Live camera access is limited in browser environments.")
            st.info("To use live camera detection, please run this app locally.")
            return
        
        # For local execution
        st.info("Attempting to access camera... You may need to grant permissions.")
        
        # Try to access camera with proper error handling
        cap = None
        for camera_index in [0, 1, 2]:  # Try common camera indices
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                st.success(f"Connected to camera index {camera_index}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            st.error("""
            Could not access any camera. Please:
            1. Ensure a camera is connected
            2. Grant camera permissions if prompted
            3. Try refreshing the page
            """)
            return
        
        stframe = st.empty()
        stop_button = st.button("Stop Camera")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("Couldn't read frame from camera")
                break
            
            try:
                # Run detection models
                gun_results = gun_model(frame, verbose=False)
                placard_results = placard_model(frame, verbose=False)
                behavior_results = behavior_model(frame, verbose=False)
                
                # Draw results
                annotated_frame = draw_combined_results(frame, gun_results, placard_results, behavior_results)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing frame: {e}")
                break
        
        # Cleanup
        if cap.isOpened():
            cap.release()
        stframe.empty()
        st.success("Camera feed stopped")
        
    except Exception as e:
        st.error(f"Camera error: {str(e)}")

# Streamlit app
st.title("Crowd Behaviour Analysis System")
st.subheader("Weapon Placards & Behaviour Detection using YOLOv8")
st.write("This app uses YOLOv8 models to detect weapons, placards, sticks, and fights in images, videos, and live camera feed.")
st.write("Upload an image or video to detect weapons, placards, sticks, and fights using YOLO models.")

# Initialize session state variables
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False
    st.session_state.image_path = None
    st.session_state.annotated_img = None
    st.session_state.image_file = None

if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
    st.session_state.video_path = None
    st.session_state.frame_count = 0
    st.session_state.total_frames = 0

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real Time Detection"])

with tab1:
    st.header("Image Detection")
    
    if not st.session_state.image_processed:
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="image_uploader")
        if image_file is not None:
            st.session_state.image_file = image_file
            annotated_img, output_path = detect_image(image_file)
            if annotated_img is not None:
                st.session_state.annotated_img = annotated_img
                st.session_state.image_path = output_path
                st.session_state.image_processed = True
                st.rerun()
    
    if st.session_state.image_processed and st.session_state.annotated_img is not None:
        annotated_img_rgb = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption="Detected Image", use_container_width=True)
        
        with open(st.session_state.image_path, "rb") as file:
            btn = st.download_button(
                label="Download Detected Image",
                data=file,
                file_name=os.path.basename(st.session_state.image_path),
                mime="image/jpeg",
                key="image_download_button"
            )
            if btn:
                st.success("Download started!")
        
        if st.button("Process New Image"):
            st.session_state.image_processed = False
            st.session_state.image_path = None
            st.session_state.annotated_img = None
            st.session_state.image_file = None
            st.rerun()

with tab2:
    st.header("Video Detection")
    
    if not st.session_state.video_processed:
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video_uploader")
        if video_file is not None:
            output_path = detect_video(video_file)
            if output_path:
                st.session_state.video_path = output_path
                st.session_state.video_processed = True
                st.rerun()
    
    if st.session_state.video_processed and st.session_state.video_path:
        try:
            with open(st.session_state.video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write(f"Total frames processed: {st.session_state.frame_count}/{st.session_state.total_frames}")
            st.write(f"Video file size: {os.path.getsize(st.session_state.video_path) / (1024*1024):.2f} MB")
            st.download_button(
                label="Download Detected Video",
                data=video_bytes,
                file_name=os.path.basename(st.session_state.video_path),
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error displaying video: {e}")
            st.write("Try downloading it")
        
        if st.button("Process New Video"):
            st.session_state.video_processed = False
            st.session_state.video_path = None
            st.session_state.frame_count = 0
            st.session_state.total_frames = 0
            st.rerun()

with tab3:
    st.header("Live Camera Detection")
    
    if st.button("Start Live Detection"):
        detect_live_camera()
    
    if st.button("Stop Live Detection"):
        st.session_state.camera_active = False
        st.rerun()  # Refresh to clear the camera feed
