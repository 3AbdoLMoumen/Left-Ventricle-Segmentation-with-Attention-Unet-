import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = "model_unet_LV.keras"
MODEL_IMG_SIZE = (128, 128)  
VIDEO_SOURCE = "example_video.avi"
ALPHA = 0.3                   # Opacity of the mask overlay (0.0 = invisible, 1.0 = solid)

# --- SMOOTHING CONFIG ---
THRESHOLD = 0.3
MORPH_KERNEL_SIZE = 5         # bigger = more aggressive smoothing/hole filling
TEMPORAL_SMOOTHING = True     # blend mask probability across frames to reduce flicker
TEMPORAL_ALPHA = 0.6          # weight of current frame vs. previous (0-1, higher = less smoothing)
KEEP_LARGEST_ONLY = True      # discard small speckle blobs, keep only main LV blob

print("Loading Keras U-Net model...")
model = tf.keras.models.load_model(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    exit()

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

# holds the previous frame's smoothed probability map (at model resolution) for temporal blending
prev_prob = None

print("Starting live echo stream. Press 'q' to exit.")
frame_idx = 0
while True:
    ret, frame = cap.read()
    frame_idx+=1
    if not ret:
        print("End of video stream or failed to grab frame.")
        break
    if frame_idx%2:
        continue
    orig_h, orig_w = frame.shape[:2]

    # 1. Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, MODEL_IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

    # 2. Model prediction -> raw probability map (still at MODEL_IMG_SIZE, values 0-1)
    prob = model.predict(input_tensor, verbose=0)[0, ..., 0]

    # 3. Temporal smoothing: exponential moving average of the *probability* map
    #    This removes frame-to-frame flicker/jitter in the mask boundary.
    if TEMPORAL_SMOOTHING:
        if prev_prob is None:
            prev_prob = prob 
        else:
            prob = TEMPORAL_ALPHA * prob + (1 - TEMPORAL_ALPHA) * prev_prob
            prev_prob = prob

    # 4. Upsample the *probability map* 
    prob_resized = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 5. Slight Gaussian blur on the probability map before thresholding smooths the
    #    eventual contour without needing to touch the binary mask itself.
    prob_resized = cv2.GaussianBlur(prob_resized, (7, 7), 0)

    # 6. Threshold to get the binary mask at full resolution
    mask_resized = (prob_resized > THRESHOLD).astype(np.uint8) * 255

    # 7. Morphological cleanup: opening removes small speckle noise, closing fills
    #    small holes, both smooth the boundary further.
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, morph_kernel)
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, morph_kernel)

    # 8. Keep only the largest connected component to drop stray blobs
    if KEEP_LARGEST_ONLY:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_resized, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_resized = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # 9. Build the overlay
    overlay_color = np.zeros_like(frame)
    overlay_color[mask_resized > 0] = [0, 0, 255]
    blended_frame = cv2.addWeighted(overlay_color, ALPHA, frame, 1.0, 0)

    # 10. Smooth contour outline using approxPolyDP to reduce pixel-level jaggedness
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_contours = []
    for c in contours:
        if cv2.contourArea(c) < 20:
            continue
        epsilon = 0.002 * cv2.arcLength(c, True)
        smoothed_contours.append(cv2.approxPolyDP(c, epsilon, True))
    cv2.drawContours(blended_frame, smoothed_contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    blended_frame = cv2.resize(blended_frame, (640, 480))

    # HUD text
    cv2.putText(blended_frame, "LV SEGMENTATION ACTIVE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    lv_pixel_area = np.sum(mask_resized > 0)
    cv2.putText(blended_frame, f"LV Area Prox: {lv_pixel_area} px", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Echo Live AI HUD", blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()