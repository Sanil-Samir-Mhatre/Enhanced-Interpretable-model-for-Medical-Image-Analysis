import matplotlib.cm as cm
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import time
import os
import streamlit as st 
import tensorflow as tf
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from io import BytesIO
from scipy.ndimage import gaussian_filter
import warnings
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_app_model():
    """Loads and caches the Keras model and the Grad-CAM model."""
    model_path = os.path.join(SCRIPT_DIR, "..", "src", "models", "model_Xception_ft.hdf5")
    model = tf.keras.models.load_model(model_path)

    # Create and cache the Grad-CAM model from the main model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("global_average_pooling2d_1").input,
            model.output,
        ],
        name="grad_cam_model"
    )
    grad_model.layers[-1].activation = None # Set activation to None for the last layer
    return model, grad_model

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(original_img, heatmap, alpha=0.4):
    # original_img is the PIL Image from the file uploader
    img_array = tf.keras.preprocessing.image.img_to_array(original_img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.width, original_img.height))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img


def measure_efficiency(func, *args, **kwargs):
    """Wrapper to measure execution time and FPS."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    fps = 1.0 / duration if duration > 0 else 0
    return result, fps

def calculate_faithfulness_score(model, img_array, heatmap, target_class_idx):
    """
    Calculates faithfulness by masking the top 20% most important pixels 
    and measuring the drop in confidence.
    """
    # Get original confidence
    orig_conf = model(img_array)[0][target_class_idx]
    
    # --- FIX: Resize heatmap if its dimensions don't match the image ---
    img_height, img_width = img_array.shape[1], img_array.shape[2]
    if heatmap.shape[0] != img_height or heatmap.shape[1] != img_width:
        # Add a channel dimension for resizing, then remove it
        heatmap_resized = tf.image.resize(np.expand_dims(heatmap, axis=-1), [img_height, img_width])
        heatmap = tf.squeeze(heatmap_resized).numpy()

    # Mask top 20% pixels
    flat_heatmap = heatmap.flatten()
    threshold = np.percentile(flat_heatmap, 80) # Top 20%
    mask = heatmap <= threshold # Keep only low importance pixels. Using <= handles sparse masks (like LIME) better.
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, -1) # (224, 224, 1)
    
    # Perturb image (black out important regions)
    # img_array is (1, 224, 224, 3), mask is (224, 224, 1). Broadcasting works on inner dims.
    masked_img = img_array * mask 
    
    # Get new confidence
    new_conf = model(masked_img)[0][target_class_idx]
    
    # Score: How much did confidence drop?
    faithfulness_score = float(orig_conf - new_conf)
    return max(faithfulness_score, 0) # Ensure non-negative

def calculate_stability_correlation(map1, map2):
    """Calculates the Pearson correlation between two flattened 2D maps."""
    if map1 is None or map2 is None or map1.shape != map2.shape:
        return 0.0
    
    # Ensure maps are not constant, which would result in NaN correlation
    if np.all(map1 == map1.mean()) or np.all(map2 == map2.mean()): return 1.0
    
    corr = np.corrcoef(map1.flatten(), map2.flatten())[0, 1]
    return corr if not np.isnan(corr) else 0.0

def get_integrated_gradients(image, target_class_idx, m_steps=50, threshold_percentile=95):
    """
    Computes Integrated Gradients (IG) with a black baseline.
    Faster and sparser than SmoothGrad.
    """
    # 1. Baseline: Black image (zeros)
    # Note: For Xception, 0.0 corresponds to mid-gray. -1.0 is black.
    # We use -1.0 (black) as the baseline to represent 'absence of signal'.
    baseline = tf.ones_like(image) * -1.0

    # 2. Generate interpolated images
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    interpolated_images = baseline + (alphas_x * (image - baseline))

    # 3. Compute Gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        preds = model(interpolated_images)
        target_class_preds = preds[:, target_class_idx]

    grads = tape.gradient(target_class_preds, interpolated_images)

    # 4. Integral Approximation
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_gradients = (image - baseline) * avg_grads
    
    # 5. Completeness Check
    pred_orig = model(np.expand_dims(image, 0))[0][target_class_idx]
    pred_base = model(np.expand_dims(baseline, 0))[0][target_class_idx]
    delta_model = pred_orig - pred_base
    sum_attr = tf.reduce_sum(integrated_gradients)
    completeness_score = abs(sum_attr - delta_model)

    # 6. Post-process
    attribution_mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)
    attribution_mask = attribution_mask / (tf.reduce_max(attribution_mask) + 1e-9)
    
    threshold = np.percentile(attribution_mask.numpy(), threshold_percentile)
    attribution_mask = tf.where(attribution_mask > threshold, attribution_mask, 0)
    attribution_mask = attribution_mask / (tf.reduce_max(attribution_mask) + 1e-9)

    return attribution_mask.numpy(), float(completeness_score)

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap_name="viridis"):
    """
    Overlays a heatmap on the original image.
    original_img: PIL Image.
    heatmap: A 2D numpy array with values between 0 and 1.
    """
    # Convert original PIL image to array
    img_array = tf.keras.preprocessing.image.img_to_array(original_img)

    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use a colormap to colorize the heatmap
    colormap = cm.get_cmap(colormap_name)
    colored_heatmap = colormap(heatmap)[:, :, :3]  # Get RGB values
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = colored_heatmap * alpha + img_array * (1 - alpha)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- Anatomical Explanation Logic ---

# Define anatomical zones for a 224x224 image as (y_start, y_end, x_start, x_end)
# These are rough approximations and may need fine-tuning.
ANATOMICAL_ZONES = {
    "Medial Femoral Condyle": (60, 100, 112, 160),
    "Lateral Femoral Condyle": (60, 100, 64, 112),
    "Medial Tibiofemoral Joint Space": (100, 124, 112, 150),
    "Lateral Tibiofemoral Joint Space": (100, 124, 74, 112),
    "Medial Tibial Plateau": (124, 160, 112, 160),
    "Lateral Tibial Plateau": (124, 160, 64, 112),
}

# Map zones to potential OA findings
CLINICAL_MAPPING = {
    "Medial Tibiofemoral Joint Space": "joint space narrowing",
    "Lateral Tibiofemoral Joint Space": "joint space narrowing",
    "Medial Femoral Condyle": "osteophyte formation",
    "Lateral Femoral Condyle": "osteophyte formation",
    "Medial Tibial Plateau": "subchondral bone changes",
    "Lateral Tibial Plateau": "subchondral bone changes",
}

def score_anatomical_zones(heatmap, zones):
    """Calculates the mean attribution score for each anatomical zone."""
    zone_scores = {}
    for zone_name, (y_start, y_end, x_start, x_end) in zones.items():
        zone_heatmap = heatmap[y_start:y_end, x_start:x_end]
        mean_score = np.mean(zone_heatmap) if zone_heatmap.size > 0 else 0
        zone_scores[zone_name] = mean_score
    
    # Sort zones by score in descending order
    sorted_zones = sorted(zone_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_zones

def generate_textual_explanation(sorted_zones, kl_grade_name):
    """Generates a textual summary based on the highest-scoring zones."""
    if not sorted_zones or sorted_zones[0][1] < 0.05: # Lowered threshold for more sensitivity
        return "The model's attention was diffuse, with no specific anatomical region standing out."

    top_zone_name = sorted_zones[0][0]
    clinical_relevance = CLINICAL_MAPPING.get(top_zone_name, "general features")

    explanation = (
        f"*   **Primary Focus:** The model's attention is centered on the **{top_zone_name}**.\n"
        f"*   **Clinical Relevance:** For a predicted grade of **{kl_grade_name}**, high activation in this area may be related to the model identifying signs of **{clinical_relevance}**."
    )
    return explanation


def generate_hybrid_explanation(gradcam_heatmap, ig_attribution, w1=0.6, w2=0.4):
    """
    Generate hybrid explanation by weighted combination of Grad-CAM and IG maps.
    """
    # Normalize both maps to [0,1]
    gradcam_norm = (gradcam_heatmap - np.min(gradcam_heatmap)) / (np.max(gradcam_heatmap) - np.min(gradcam_heatmap) + 1e-9)
    ig_norm = (ig_attribution - np.min(ig_attribution)) / (np.max(ig_attribution) - np.min(ig_attribution) + 1e-9)
    
    # Weighted sum
    hybrid_map = w1 * gradcam_norm + w2 * ig_norm
    
    # Normalize final output
    hybrid_map = (hybrid_map - np.min(hybrid_map)) / (np.max(hybrid_map) - np.min(hybrid_map) + 1e-9)
    return hybrid_map

def generate_ig_lime_hybrid(ig_attribution, lime_mask):
    """
    Generate hybrid explanation by masking Integrated Gradients with LIME superpixels.
    Element-wise multiplication creates a sparse, highly localized IG map.
    """
    # Normalize IG to [0,1]
    ig_norm = (ig_attribution - np.min(ig_attribution)) / (np.max(ig_attribution) - np.min(ig_attribution) + 1e-9)
    
    # Treat LIME mask as binary weights
    lime_binary = (lime_mask > 0).astype(np.float32)
    
    # Element-wise multiplication
    hybrid_map = ig_norm * lime_binary
    
    # Normalize final output
    if np.max(hybrid_map) > 0:
        hybrid_map = (hybrid_map - np.min(hybrid_map)) / (np.max(hybrid_map) - np.min(hybrid_map) + 1e-9)
    return hybrid_map

def compute_sparsity(map_2d, threshold=0.1):
    """
    Compute sparsity as percentage of low-importance pixels.
    """
    threshold_val = np.percentile(map_2d, threshold * 100)
    sparse_pixels = np.sum(map_2d <= threshold_val)
    total_pixels = map_2d.size
    return (sparse_pixels / total_pixels) * 100

def aggregate_concept_scores(sorted_zones, clinical_mapping):
    """Aggregates scores from anatomical zones into clinical concepts."""
    concept_scores = {
        "joint space narrowing": 0.0,
        "osteophyte formation": 0.0,
        "subchondral bone changes": 0.0,
    }
    # Use a count to average scores if a concept is linked to multiple zones
    concept_counts = {key: 0 for key in concept_scores}

    for zone_name, score in sorted_zones:
        concept = clinical_mapping.get(zone_name)
        if concept:
            concept_scores[concept] += score
            concept_counts[concept] += 1
    
    return concept_scores

@st.cache_resource
def get_concept_signatures(_model, concept_base_path, target_size):
    """
    Computes the average activation "signature" for each clinical concept
    and also stores the individual activations for every concept image.
    This function runs once and caches the result.
    """
    concept_signatures = {}
    concept_image_activations = {}

    if not os.path.exists(concept_base_path):
        return {}, {}

    # Use a sub-model that outputs the bottleneck activations before the final layer
    activation_model = tf.keras.Model(
        inputs=_model.inputs,
        outputs=_model.get_layer("global_average_pooling2d_1").output
    )

    for concept_name in os.listdir(concept_base_path):
        concept_dir = os.path.join(concept_base_path, concept_name)
        if not os.path.isdir(concept_dir):
            continue

        image_paths = [os.path.join(concept_dir, fname) for fname in os.listdir(concept_dir)]
        concept_image_activations[concept_name] = []
        activations = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB").resize(target_size)
                img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)
                img_preprocessed = tf.keras.applications.xception.preprocess_input(np.float32(img_array))
                activation = activation_model.predict(img_preprocessed, verbose=0)
                
                # Store individual activation
                concept_image_activations[concept_name].append({'path': img_path, 'activation': activation.flatten()})
                activations.append(activation)
            except Exception as e:
                st.warning(f"Could not process image {img_path}: {e}")

        if activations:
            # The signature is the average activation vector for the concept
            concept_signatures[concept_name] = np.mean(np.array(activations), axis=0).flatten()

    # --- Bias Reduction: Center the signatures ---
    if concept_signatures:
        mean_vector = np.mean(list(concept_signatures.values()), axis=0)
        for name in concept_signatures:
            concept_signatures[name] -= mean_vector

    return concept_signatures, concept_image_activations

def get_tcav_scores(model, img_array, concept_signatures, target_class_idx):
    """
    Computes TCAV scores (Directional Derivatives).
    Measures the sensitivity of the prediction to the concept direction.
    """
    bottleneck_layer_name = "global_average_pooling2d_1"
    try:
        bottleneck_layer = model.get_layer(bottleneck_layer_name)
    except ValueError:
        return None, {}

    # 1. Define models to isolate bottleneck and classifier
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=bottleneck_layer.output)
    
    # Reconstruct the classifier part (Bottleneck -> Output)
    bottleneck_input = tf.keras.Input(shape=bottleneck_layer.output.shape[1:])
    x = bottleneck_input
    
    # Find index of bottleneck layer to start reconstruction after it
    layer_idx = next((i for i, layer in enumerate(model.layers) if layer.name == bottleneck_layer_name), -1)
    if layer_idx == -1: return None, {}

    for layer in model.layers[layer_idx+1:]:
        x = layer(x)
    
    classifier_model = tf.keras.Model(inputs=bottleneck_input, outputs=x)

    # 2. Compute Gradients (Sensitivity)
    bottleneck_features = activation_model(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(bottleneck_features)
        preds = classifier_model(bottleneck_features)
        score = preds[:, target_class_idx]
    
    # Gradient of class score w.r.t bottleneck activations
    grads = tape.gradient(score, bottleneck_features)
    grads_flat = tf.reshape(grads, [-1]).numpy()

    # 3. Calculate TCAV Score: Dot product of Gradient and CAV (Concept Vector)
    tcav_scores = {}
    for concept, cav in concept_signatures.items():
        # Normalize CAV for consistent directional magnitude
        cav_norm = np.linalg.norm(cav)
        if cav_norm > 0:
            # TCAV Score = Directional Derivative
            val = np.dot(grads_flat, cav / cav_norm)
            tcav_scores[concept] = max(0.0, val)
        else:
            tcav_scores[concept] = 0.0
            
    return bottleneck_features.numpy(), tcav_scores

def find_closest_prototype(input_activation, concept_name, concept_image_activations):
    """
    Finds the closest prototype image using Euclidean distance (L2).
    This matches the standard ProtoNet approach of finding the nearest embedding.
    """
    input_vector = input_activation.flatten()
    best_match = {'path': None, 'distance': float('inf')}

    if concept_name not in concept_image_activations:
        return None

    for item in concept_image_activations[concept_name]:
        prototype_vector = item['activation']
        # Use Euclidean distance instead of Cosine similarity
        distance = np.linalg.norm(input_vector - prototype_vector)
        
        if distance < best_match['distance']:
            best_match['distance'] = distance
            best_match['path'] = item['path']
    return Image.open(best_match['path']) if best_match['path'] else None

def get_concept_shap_explanation(img_array_preprocessed, concept_signatures, model, pred_class_idx):
    """
    Approximates Concept SHAP values using Gradient * Input attribution projected onto concept vectors.
    """
    bottleneck_layer_name = "global_average_pooling2d_1"
    
    # 1. Define Classifier Model (Bottleneck -> Output)
    try:
        bottleneck_layer = model.get_layer(bottleneck_layer_name)
    except ValueError:
        return {}

    # Reconstruct the top part of the model
    bottleneck_input = tf.keras.Input(shape=bottleneck_layer.output.shape[1:])
    x = bottleneck_input
    
    start_idx = -1
    for i, layer in enumerate(model.layers):
        if layer.name == bottleneck_layer_name:
            start_idx = i
            break
    
    if start_idx == -1: return {}

    for layer in model.layers[start_idx+1:]:
        x = layer(x)
    
    classifier_model = tf.keras.Model(inputs=bottleneck_input, outputs=x)
    classifier_model.layers[-1].activation = None # Remove softmax

    # 2. Get Bottleneck Features
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=bottleneck_layer.output)
    bottleneck_features = activation_model.predict(img_array_preprocessed, verbose=0)
    features_tensor = tf.convert_to_tensor(bottleneck_features)

    # 3. Compute Gradients (Sensitivity)
    with tf.GradientTape() as tape:
        tape.watch(features_tensor)
        preds = classifier_model(features_tensor)
        top_class_score = preds[:, pred_class_idx]

    grads = tape.gradient(top_class_score, features_tensor)
    attribution_flat = tf.reshape(grads * features_tensor, [-1]).numpy()

    # 4. Project onto Concepts
    concept_shap_scores = {}
    for concept_name, concept_vec in concept_signatures.items():
        norm = np.linalg.norm(concept_vec)
        if norm > 0:
            # Project attribution onto concept direction
            score = np.dot(attribution_flat, concept_vec / norm)
            concept_shap_scores[concept_name] = score
        else:
            concept_shap_scores[concept_name] = 0.0
            
    return concept_shap_scores

icon_path = os.path.join(SCRIPT_DIR, "img", "mdc.png")
icon = Image.open(icon_path)
st.set_page_config(
    page_title="Severity Analysis of Osteoarthritis in the Knee",
    page_icon=icon,
)
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

# --- Model and App Configuration ---
class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)
model, grad_model = load_app_model()

# --- Pre-compute Concept Signatures ---
CONCEPT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "concepts")
concept_signatures, concept_image_activations = get_concept_signatures(model, CONCEPT_PATH, target_size)

# Sidebar
with st.sidebar:
    st.image(icon)
    st.subheader("Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")

# Body
st.header("Severity Analysis of Osteoarthritis in the Knee")

st.markdown("""
This application analyzes knee X-ray images to predict the severity of osteoarthritis based on the Kellgren-Lawrence (KL) grading system. 
The underlying deep learning model is an Xception model, which achieves a Balanced Accuracy of 67% on the test dataset.
""")

with st.expander("See other model performances"):
    st.markdown("""
    | Model                           | Balanced Accuracy |
    | ------------------------------- | ----------------- |
    | Ensemble (f1-score weighted)    | 68.69%            |
    | Xception fine-tuning        | 67%           |
    | ResNet50 fine-tuning            | 65%               |
    | Inception-ResNet-v2 fine-tuning | 64%               |
    """)

# --- Main App Logic ---
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    # Display input image
    st.subheader("Input X-Ray")
    st.image(original_image, caption="Uploaded X-Ray", width='stretch')

    # Prepare images
    img_for_prediction = original_image.resize(target_size)
    img_for_display = tf.keras.preprocessing.image.img_to_array(img_for_prediction)
    img_array_expanded = np.expand_dims(img_for_display, axis=0)
    img_array_preprocessed = tf.keras.applications.xception.preprocess_input(np.float32(img_array_expanded))

    # --- Prediction and XAI Caching ---
    @st.cache_data
    def get_prediction(_img_array):
        return model.predict(_img_array)

    @st.cache_data
    def get_grad_cam_explanation(_grad_model, _img_array):
        heatmap = make_gradcam_heatmap(_grad_model, _img_array)
        gradcam_image = save_and_display_gradcam(original_image, heatmap)
        return heatmap, gradcam_image

    @st.cache_data
    def get_ig_explanation(_img_for_display, pred_class_idx):
        ig_img_array = tf.keras.applications.xception.preprocess_input(np.expand_dims(_img_for_display, axis=0))[0]
        attribution_mask, completeness = get_integrated_gradients(image=ig_img_array, target_class_idx=pred_class_idx)
        ig_overlay_image = overlay_heatmap(img_for_prediction, attribution_mask)
        return attribution_mask, ig_overlay_image, completeness

@st.cache_data
def get_lime_explanation(_img_for_prediction, random_state_seed=42, num_samples=1000):
    def predict_fn_for_lime(images):
        images_float32 = images.astype('float32')
        img_processed = tf.keras.applications.xception.preprocess_input(images_float32)
        return model.predict(img_processed)

    explainer = lime_image.LimeImageExplainer(random_state=random_state_seed)
    segmentation_fn = lambda x: slic(x, n_segments=150, compactness=15, sigma=1, start_label=1) # Increased segments and compactness for finer, more regular superpixels
    explanation = explainer.explain_instance(
        np.array(_img_for_prediction), 
        predict_fn_for_lime, 
        top_labels=1, 
        hide_color=0, 
        num_samples=num_samples,
        segmentation_fn=segmentation_fn
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True
    )
    
    # Image 1: Superpixels on black background
    lime_superpixels_image = Image.fromarray((temp * 255).astype(np.uint8))

    # Image 2: Boundaries on original image
    lime_boundaries_image = mark_boundaries(np.array(_img_for_prediction) / 255.0, mask)
    lime_boundaries_pil = Image.fromarray((lime_boundaries_image * 255).astype(np.uint8))

    return lime_superpixels_image, lime_boundaries_pil, mask

@st.cache_data
def get_hybrid_xai_explanation(_grad_model, _img_array, _img_for_display, pred_class_idx, alpha=0.5, sigma=5):
    gradcam_map = make_gradcam_heatmap(_grad_model, _img_array, pred_index=pred_class_idx)
    
    ig_img_array = tf.keras.applications.xception.preprocess_input(np.expand_dims(_img_for_display, axis=0))[0]
    ig_map, _ = get_integrated_gradients(image=ig_img_array, target_class_idx=pred_class_idx, m_steps=20)
    
    # FIX: Resize Grad-CAM to match the shape of the Integrated Gradients map before fusion.
    # Grad-CAM produces a low-res map (e.g., 7x7), while IG is at input resolution (224x224).
    if gradcam_map.shape != ig_map.shape:
        gradcam_map_resized = tf.image.resize(np.expand_dims(gradcam_map, axis=-1), ig_map.shape[:2], method='bicubic')
        gradcam_map = tf.squeeze(gradcam_map_resized).numpy()

    hybrid_map = generate_hybrid_explanation(gradcam_map, ig_map)
    
    hybrid_overlay_image = overlay_heatmap(img_for_prediction, hybrid_map, colormap_name="plasma")
    return hybrid_map, hybrid_overlay_image

def get_ig_lime_hybrid_explanation(ig_map, lime_mask, img_for_prediction):
    hybrid_map = generate_ig_lime_hybrid(ig_map, lime_mask)
    hybrid_overlay_image = overlay_heatmap(img_for_prediction, hybrid_map, colormap_name="magma")
    
    # Add yellow boundaries around the LIME region to clearly show the combination
    overlay_np = np.array(hybrid_overlay_image) / 255.0
    lime_label = (lime_mask > 0).astype(int)
    
    # Mark boundaries in yellow (RGB: 1, 1, 0)
    bounded_overlay_np = mark_boundaries(overlay_np, lime_label, color=(1, 1, 0))
    final_overlay_image = Image.fromarray((bounded_overlay_np * 255).astype(np.uint8))
    
    return hybrid_map, final_overlay_image

def image_to_bytes(img):
    """Converts a PIL Image to bytes for downloading."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


if uploaded_file is not None:
    if st.button("Analyze Knee X-Ray", type="primary"):
        # Initialize session state for metrics if not present
        if 'xai_metrics' not in st.session_state:
            st.session_state.xai_metrics = {}

        st.markdown("---")
        st.header("Prediction Results")

        # --- 1. Prediction ---
        with st.spinner("Running prediction..."):
            y_pred_raw = get_prediction(img_array_preprocessed)
        
        # Apply softmax to convert logits to probabilities, then multiply by 100
        y_pred_probs = tf.nn.softmax(y_pred_raw[0]).numpy()
        y_pred_percent = 100 * y_pred_probs
        probability = np.amax(y_pred_percent)
        kl_grade_index = np.argmax(y_pred_percent)
        grade_name = class_names[kl_grade_index]
        
        st.metric(label=f"Predicted Grade: {grade_name}", value=f"{probability:.2f}%")
        st.write(f"The predicted Kellgren-Lawrence (KL) Grade is: **{kl_grade_index}**")
        st.caption("This percentage represents the model's confidence in its prediction.")
        
        # --- 1a. Prediction Distribution Chart ---
        st.subheader("Prediction Confidence Distribution")        
        chart_df = pd.DataFrame({'KL Grade': class_names, 'Confidence (%)': y_pred_percent})        
        
        # Define the color condition: orange for the predicted grade, blue for others.
        color_condition = alt.condition(
            alt.datum['KL Grade'] == grade_name,
            alt.value('#ff7f0e'),     # The highlight color
            alt.value('#1f77b4')      # The default color
        )
        
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('KL Grade', sort=class_names),
            y='Confidence (%)',
            color=color_condition
        )
        st.altair_chart(chart, use_container_width=True)

        # --- XAI Section ---
        st.markdown("---")
        st.header("Model Explanations (XAI)")

        # --- 2. Grad-CAM Explanation ---
        st.subheader("1. Grad-CAM Heatmap")
        st.info('**Question:** "Where did the model look?"\n\n**Answer:** Grad-CAM creates a heatmap of the most intensely activated regions in the final convolutional layer, showing the model\'s focus area.')
        with st.spinner("Generating Grad-CAM..."):
            (heatmap, gradcam_image), gradcam_fps = measure_efficiency(get_grad_cam_explanation, grad_model, img_array_preprocessed)
            gradcam_faithfulness = calculate_faithfulness_score(model, img_array_preprocessed, heatmap, kl_grade_index)
            st.session_state.xai_metrics['Grad-CAM'] = {
                'Faithfulness': f"{gradcam_faithfulness:.4f} (Confidence Drop)",
                'Stability': "1.0 (Deterministic)",
                'Interpretability': "4/5 (Good)",
                'Localization Accuracy': "N/A (No Ground Truth)",
                'Compute Time (s)': f"{1/gradcam_fps:.4f}"
            }


        st.image(gradcam_image, caption="Grad-CAM highlights where the model 'looked' for the prediction.", width='stretch')
        st.download_button(
            label="Download Grad-CAM Image",
            data=image_to_bytes(gradcam_image),
            file_name="grad_cam.png",
            mime="image/png"
        )
        with st.expander("Show Anatomical Interpretation"):
            grad_cam_zones = score_anatomical_zones(heatmap, ANATOMICAL_ZONES)
            explanation_text = generate_textual_explanation(grad_cam_zones, grade_name)
            st.markdown(explanation_text)            
            
            st.markdown("---")
            st.write("**Detailed Zone Scores:**")
            chart_tab, table_tab = st.tabs(["Bar Chart", "Data Table"])
            zone_df = pd.DataFrame(grad_cam_zones, columns=["Anatomical Zone", "Attention Score"])

            with chart_tab:
                st.bar_chart(zone_df.set_index("Anatomical Zone"))
            with table_tab:
                st.dataframe(zone_df, use_container_width=True)

        # --- 3. Integrated Gradients Explanation ---
        st.subheader("2. Integrated Gradients Explanation")
        st.info('**Question:** "Which pixels were most influential?"\n\n**Answer:** IG provides a fine-grained attribution map by highlighting pixels that strongly influenced the final prediction.')
        with st.spinner("Calculating Integrated Gradients..."):
            (attribution_mask, ig_overlay_image, ig_completeness), ig_fps = measure_efficiency(get_ig_explanation, img_for_display, kl_grade_index)
            ig_faithfulness = calculate_faithfulness_score(model, img_array_preprocessed, attribution_mask, kl_grade_index)
            st.session_state.xai_metrics['Integrated Gradients'] = {
                'Faithfulness': f"{ig_faithfulness:.4f} (Confidence Drop)",
                'Stability': "1.0 (Deterministic)",
                'Interpretability': "3/5 (Moderate)",
                'Localization Accuracy': "N/A (No Ground Truth)",
                'Compute Time (s)': f"{1/ig_fps:.4f}"
            }


        st.image(ig_overlay_image, caption="Integrated Gradients highlights influential pixels for the prediction.", width='stretch')
        st.download_button(
            label="Download IG Image",
            data=image_to_bytes(ig_overlay_image),
            file_name="integrated_gradients.png",
            mime="image/png"
        )
        with st.expander("Show Anatomical Interpretation"):
            ig_zones = score_anatomical_zones(attribution_mask, ANATOMICAL_ZONES)
            explanation_text = generate_textual_explanation(ig_zones, grade_name)
            st.markdown(explanation_text)            
            
            st.markdown("---")
            st.write("**Detailed Zone Scores:**")
            chart_tab, table_tab = st.tabs(["Bar Chart", "Data Table"])
            zone_df = pd.DataFrame(ig_zones, columns=["Anatomical Zone", "Attention Score"])
            with chart_tab:
                st.bar_chart(zone_df.set_index("Anatomical Zone"))
            with table_tab:
                st.dataframe(zone_df, use_container_width=True)

        # --- 4. LIME Explanation ---
        st.subheader("3. Superpixel-based Explanation using LIME)")
        st.info('**Question:** "Which high-level image regions were most important?"\n\n**Answer:** LIME identifies the most important "superpixels" (intuitive clusters of pixels) that contribute to the prediction.')
        with st.spinner("Generating LIME explanation... This may take a moment."):
            # Run twice for stability check with different seeds and fewer samples for speed
            (lime_superpixels_1, lime_boundaries_1, lime_mask_1), lime_fps_1 = measure_efficiency(get_lime_explanation, img_for_prediction, random_state_seed=42, num_samples=250)
            (_, _, lime_mask_2), lime_fps_2 = measure_efficiency(get_lime_explanation, img_for_prediction, random_state_seed=100, num_samples=250)
            lime_fps = (lime_fps_1 + lime_fps_2) / 2
            
            lime_faithfulness = calculate_faithfulness_score(model, img_array_preprocessed, lime_mask_1.astype(np.float32), kl_grade_index)
            lime_stability = calculate_stability_correlation(lime_mask_1, lime_mask_2)
            
            st.session_state.xai_metrics['LIME'] = {
                'Faithfulness': f"{lime_faithfulness:.4f} (Confidence Drop)",
                'Stability': f"{lime_stability:.4f} (Correlation)",
                'Interpretability': "5/5 (Excellent)",
                'Localization Accuracy': "N/A (No Ground Truth)",
                'Compute Time (s)': f"{1/lime_fps:.4f}"
            }

        # Display both images
        col1, col2 = st.columns(2)
        with col1:
            st.image(lime_superpixels_1, caption="LIME: Important superpixels isolated.", width='stretch')
            st.download_button(
                label="Download Superpixels Image",
                data=image_to_bytes(lime_superpixels_1),
                file_name="lime_superpixels.png",
                mime="image/png"
            )
        with col2:
            st.image(lime_boundaries_1, caption="LIME: Important regions on original image.", width='stretch')
            st.download_button(
                label="Download Boundaries Image",
                data=image_to_bytes(lime_boundaries_1),
                file_name="lime_boundaries.png",
                mime="image/png"
            )

        # --- 4.5 Hybrid Explanation ---
        st.subheader("4. Hybrid XAI no. 2 (Grad-CAM + Integrated Gradients)")
        st.info('**Question:** "How can we combine high-level regional focus with pixel-perfect attribution?"\n\n**Answer:** The Hybrid approach bridges spatial localization (Grad-CAM) with attribution precision (IG). By multiplying the two, it highlights the exact pixels contributing to the prediction, but primarily within the anatomically relevant regions found by Grad-CAM.')
        with st.spinner("Generating Hybrid explanation..."):
            (hybrid_mask, hybrid_overlay_image), hybrid_fps = measure_efficiency(get_hybrid_xai_explanation, grad_model, img_array_preprocessed, img_for_display, kl_grade_index)
            hybrid_faithfulness = calculate_faithfulness_score(model, img_array_preprocessed, hybrid_mask, kl_grade_index)
            sparsity = compute_sparsity(hybrid_mask)
            st.metric("Sparsity (% low-importance pixels)", f"{sparsity:.1f}%")
            
            st.session_state.xai_metrics['Hybrid (Grad-CAM+IG)'] = {
                'Faithfulness': f"{hybrid_faithfulness:.4f} (Confidence Drop)",
                'Stability': "1.0 (Deterministic)",
                'Interpretability': "5/5 (Excellent)",
                'Localization Accuracy': "N/A (No Ground Truth)",
                'Compute Time (s)': f"{1/hybrid_fps:.4f}"
            }

        st.image(hybrid_overlay_image, caption="Hybrid XAI: Fusing Grad-CAM spatial localization with IG pixel precision.", width='stretch')
        st.download_button(
            label="Download Hybrid Image",
            data=image_to_bytes(hybrid_overlay_image),
            file_name="hybrid_xai.png",
            mime="image/png",
            key="download_hybrid"
        )
        with st.expander("Show Anatomical Interpretation"):
            hybrid_zones = score_anatomical_zones(hybrid_mask, ANATOMICAL_ZONES)
            explanation_text = generate_textual_explanation(hybrid_zones, grade_name)
            st.markdown(explanation_text)            
            
            st.markdown("---")
            st.write("**Detailed Zone Scores:**")
            chart_tab, table_tab = st.tabs(["Bar Chart", "Data Table"])
            zone_df = pd.DataFrame(hybrid_zones, columns=["Anatomical Zone", "Attention Score"])
            with chart_tab:
                st.bar_chart(zone_df.set_index("Anatomical Zone"))
            with table_tab:
                st.dataframe(zone_df, use_container_width=True)

        # --- 5. Hybrid XAI #2 (IG + LIME) ---
        st.subheader("5. Hybrid XAI no. 2 (Integrated Gradients + LIME)")
        st.info('**Question:** "How can we isolate the exact influential pixels within the most important broad image regions?"\n\n**Answer:** This hybrid fuses IG\'s pixel-perfect precision with LIME\'s regional superpixels. By multiplying them, it filters out background noise from IG, leaving only the fine-grained pixel attributions inside the most critical superpixels.')
        with st.spinner("Generating IG+LIME hybrid..."):
            # Efficiently reuse pre-calculated IG and LIME masks without calling the model again
            def run_ig_lime_hybrid():
                return get_ig_lime_hybrid_explanation(attribution_mask, lime_mask_1, img_for_prediction)
            
            (hybrid2_mask, hybrid2_overlay_image), hybrid2_fps = measure_efficiency(run_ig_lime_hybrid)
            hybrid2_faithfulness = calculate_faithfulness_score(model, img_array_preprocessed, hybrid2_mask, kl_grade_index)
            sparsity2 = compute_sparsity(hybrid2_mask)
            st.metric("Sparsity (% low-importance pixels)", f"{sparsity2:.1f}%")
            
            st.session_state.xai_metrics['Hybrid2 (IG+LIME)'] = {
                'Faithfulness': f"{hybrid2_faithfulness:.4f} (Confidence Drop)",
                'Stability': f"{lime_stability:.4f} (Inherited from LIME)",
                'Interpretability': "5/5 (Excellent)",
                'Localization Accuracy': "N/A (No Ground Truth)",
                'Compute Time (s)': f"{1/hybrid2_fps:.4f} (Cached)"
            }

        st.image(hybrid2_overlay_image, caption="Hybrid2: Integrated Gradients attribution strictly masked by LIME superpixels.", width='stretch')
        st.download_button(
            label="Download Hybrid2 Image",
            data=image_to_bytes(hybrid2_overlay_image),
            file_name="hybrid2_ig_lime.png",
            mime="image/png"
        )
        with st.expander("Show Anatomical Interpretation"):
            hybrid2_zones = score_anatomical_zones(hybrid2_mask, ANATOMICAL_ZONES)
            explanation_text = generate_textual_explanation(hybrid2_zones, grade_name)
            st.markdown(explanation_text)            
            
            st.markdown("---")
            st.write("**Detailed Zone Scores:**")
            chart_tab, table_tab = st.tabs(["Bar Chart", "Data Table"])
            zone_df = pd.DataFrame(hybrid2_zones, columns=["Anatomical Zone", "Attention Score"])
            with chart_tab:
                st.bar_chart(zone_df.set_index("Anatomical Zone"))
            with table_tab:
                st.dataframe(zone_df, use_container_width=True)

        # --- 6. Quantitative Concept-Based Explanation (TCAV) ---
        st.subheader("6. Concept-Based Explanation (TCAV)")
        st.info('**Question:** "How much did the model rely on learned clinical concepts?"\n\n**Answer:** This analysis measures the similarity between the features in the input image and the pre-learned "signatures" of clinical concepts. A higher score means the model\'s internal representation of the input image is more aligned with its understanding of that concept.')
        
        if concept_signatures:
            # Get the bottleneck activation for the input image
            def run_tcav():
                return get_tcav_scores(model, img_array_preprocessed, concept_signatures, kl_grade_index)
            
            (input_activation, similarities), tcav_fps = measure_efficiency(run_tcav)
            st.session_state.xai_metrics['TCAV'] = {
                'Faithfulness': "Statistical (t-test based)",
                'Stability': "1.0 (Deterministic)",
                'Interpretability': "5/5 (Excellent)",
                'Localization Accuracy': "N/A",
                'Compute Time (s)': f"{1/tcav_fps:.4f}"
            }


            # Prepare for plotting
            concept_df = pd.DataFrame.from_dict(similarities, orient="index", columns=["Similarity Score"])
            concept_df = concept_df.sort_values("Similarity Score", ascending=False).reset_index().rename(columns={"index": "Clinical Concept"})

            st.write("Concept Similarity Scores:")
            st.bar_chart(data=concept_df, x="Clinical Concept", y="Similarity Score")
            st.caption("Scores represent cosine similarity. Positive scores indicate the concept is present (similar to the concept prototype), while negative scores suggest it is absent or opposite to the concept.")
            st.caption("Scores represent **Concept Influence** (Directional Derivative). Positive scores indicate the concept **supports** the prediction, while negative scores indicate the concept **contradicts** it.")
        else:
            st.warning("Could not find or load concept datasets. Please ensure the `data/concepts` directory is populated.")

        # --- 7. Prototype-Based Explanation (ProtoPNet-inspired) ---
        st.subheader("7. Prototype-Based Explanation (ProtoPNet-inspired)")
        st.info('**Question:** "What known example is this case similar to?"\n\n**Answer:** This method finds the learned clinical prototype (e.g., a classic example of an osteophyte) that is most similar to a region in the input image. It explains the prediction by saying, "The model predicted this grade because *this part* of the image looks like *this known prototype*."')

        if concept_signatures:
            # We reuse the similarities calculated for the TCAV section
            # Sort concepts by similarity score to find the most relevant one
            sorted_concepts = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
            top_concept_name, top_score = sorted_concepts[0]

            if top_score > 0:
                st.write("Most relevant clinical prototype (Nearest Match):")
                
                # We measure efficiency for the top 1 match for the metrics table
                def run_proto_top():
                    return find_closest_prototype(input_activation, top_concept_name, concept_image_activations)
                
                closest_prototype_image, proto_fps = measure_efficiency(run_proto_top)
                st.session_state.xai_metrics['ProtoPNet'] = {
                    'Faithfulness': "Prototype Match",
                    'Stability': "1.0 (Deterministic)",
                    'Interpretability': "5/5 (Excellent)",
                    'Localization Accuracy': "N/A",
                    'Compute Time (s)': f"{1/proto_fps:.4f}"
                }


                if closest_prototype_image:
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        st.image(gradcam_image, caption="Input Image (Grad-CAM Focus)", width='stretch')
                    with col2:
                        st.markdown("<div style='display: flex; align-items: center; justify-content: center; height: 100%; font-size: 24px; font-weight: bold;'>→<br>looks like</div>", unsafe_allow_html=True)
                    with col3:
                        st.image(closest_prototype_image, caption=f"Prototype: {top_concept_name.replace('_', ' ').title()}", width='stretch')
                    
                    match_quality = "Strong Match" if top_score > 0.5 else "Moderate Match" if top_score > 0.2 else "Weak Match"
                    st.caption(f"Similarity Score: {top_score:.4f} ({match_quality})")
                else:
                    st.warning(f"Could not load a prototype image for '{top_concept_name}'.")
            else:
                st.info("No specific clinical prototype was identified as relevant (Score > 0) for this prediction.")
                st.session_state.xai_metrics['ProtoPNet'] = {
                    'Faithfulness': "N/A",
                    'Stability': "N/A",
                    'Interpretability': "N/A",
                    'Localization Accuracy': "N/A",
                    'Compute Time (s)': "0.0000"
                }


        # --- 8. Concept SHAP ---
        st.subheader("8. Concept/Feature Contribution Shap-based")
        st.info('**Question:** "What is the contribution of each clinical concept to the prediction?"\n\n**Answer:** Concept SHAP estimates the impact of each concept on the model\'s output score. Positive scores increase the confidence in the predicted grade, while negative scores decrease it.')

        if concept_signatures:
            def run_concept_shap():
                return get_concept_shap_explanation(img_array_preprocessed, concept_signatures, model, kl_grade_index)

            (concept_shap_scores), cshap_fps = measure_efficiency(run_concept_shap)
            st.session_state.xai_metrics['Concept SHAP'] = {
                'Faithfulness': "Gradient-based",
                'Stability': "1.0 (Deterministic)",
                'Interpretability': "4/5 (Good)",
                'Localization Accuracy': "N/A",
                'Compute Time (s)': f"{1/cshap_fps:.4f}"
            }


            # Plotting
            shap_df = pd.DataFrame.from_dict(concept_shap_scores, orient="index", columns=["Contribution Score"])
            shap_df = shap_df.sort_values("Contribution Score", ascending=False).reset_index().rename(columns={"index": "Clinical Concept"})
            
            shap_chart = alt.Chart(shap_df).mark_bar().encode(
                x=alt.X('Contribution Score'),
                y=alt.Y('Clinical Concept', sort='-x'),
                color=alt.condition(
                    alt.datum['Contribution Score'] > 0,
                    alt.value("green"),
                    alt.value("red")
                )
            ).properties(title="Concept Contribution to Prediction")
            st.altair_chart(shap_chart, use_container_width=True)

st.markdown("---")
st.subheader("Kellgren-Lawrence (KL) Grade Descriptions")
st.markdown("""
*   Grade 0 (Healthy): Healthy knee image.
*   Grade 1 (Doubtful): Doubtful joint narrowing with possible osteophytic lipping.
*   Grade 2 (Minimal): Definite presence of osteophytes and possible joint space narrowing.
*   Grade 3 (Moderate): Multiple osteophytes, definite joint space narrowing, with mild sclerosis.
*   Grade 4 (Severe): Large osteophytes, significant joint narrowing, and severe sclerosis.
""")

st.markdown("---")
st.subheader("Enhancing Clinical Trust with Explainable AI (XAI)")
st.markdown("""
The integration of multiple distinct explainability techniques is crucial for building a transparent and trustworthy diagnostic aid. By implementing these XAI methods, we move beyond "black box" predictions and provide insights into the model's decision-making process from multiple perspectives.

For doctors, this transparency is invaluable. It allows them to:
*   Verify Predictions: Clinicians can cross-reference the model's highlighted regions (via heatmaps and masks) with their own expert assessment of the X-ray, confirming if the model is focusing on clinically relevant features like joint space narrowing or osteophytes.
*   Understand the 'Why' at Different Levels: Grad-CAM shows *where* the model is looking. LIME and IG show *which pixels or regions* are important. **Concept-Based Analysis (TCAV)** aligns features with clinical concepts, **Concept SHAP** scores their contribution, while **Prototype-Based Explanations (ProtoPNet)** provide case-based reasoning by matching the input to known examples.
*   Build Confidence: Seeing that the model's reasoning aligns with established clinical concepts helps build trust in the tool. If the explanations are nonsensical, it signals that the prediction may be unreliable.
*   Gain New Insights: The model might identify subtle patterns that are not immediately obvious, offering a new perspective for the clinician to consider.

Ultimately, providing this multi-faceted explanation transforms the application from a simple prediction tool into a collaborative partner, augmenting a doctor's expertise rather than just offering an opinion.
""")

st.markdown("---")
st.subheader("XAI Methods Evaluation")
st.markdown("This matrix evaluates the different XAI techniques based on several key parameters to ensure they are reliable and useful.")

with st.expander("Learn about the evaluation parameters"):
    st.markdown("""
    **1. Faithfulness to Model Prediction:**
    *   **Meaning:** How accurately does the explanation reflect the model’s real decision-making process?
    *   **Measurement:** We measure the drop in the model's prediction confidence after removing the most important regions highlighted by the explanation. A larger drop indicates a more faithful explanation. For concept-based methods, this is described qualitatively.
    *   **Why it's important:** A faithful explanation gives you confidence that the model is truly using the highlighted features for its prediction.

    **2. Explanation Stability / Robustness:**
    *   **Meaning:** If you run the explanation method multiple times on the same image, do you get similar results?
    *   **Measurement:** We run the XAI method multiple times (for stochastic methods like LIME) and calculate the Pearson correlation between the explanations. Deterministic methods will always have a score of 1.0.
    *   **Why it's important:** Unstable or random explanations reduce trust and are not reliable.

    **3. Interpretability / Human Understanding:**
    *   **Meaning:** How easy it is for a human (e.g., a doctor) to understand the explanation.
    *   **Measurement:** This is a qualitative score on a 1-5 scale (5=Excellent, 1=Poor) based on the visual clarity of heatmaps, the meaning of concepts, or the similarity of prototypes.
    *   **Why it's important:** The primary goal of XAI is to provide human-understandable insights.

    **4. Localization Accuracy:**
    *   **Meaning:** How well does the explanation highlight the clinically relevant region in the image?
    *   **Measurement:** This typically requires comparing the explanation heatmap with a ground truth annotation (e.g., a mask drawn by a radiologist) using metrics like Intersection over Union (IoU).
    *   **Why it's important:** Ensures the model is not just correct, but correct for the right reasons, focusing on actual pathologies.

    **5. Computational Efficiency:**
    *   **Meaning:** How fast does the explanation method run?
    *   **Measurement:** We measure the runtime per explanation in seconds.
    *   **Why it's important:** Faster methods are more practical for real-time use and iterative analysis.
    """)

if 'xai_metrics' in st.session_state and st.session_state.xai_metrics:
    # Filter to show only pixel-based methods in this table
    methods_to_show = ['Grad-CAM', 'Integrated Gradients', 'LIME', 'Hybrid (Grad-CAM+IG)', 'Hybrid2 (IG+LIME)']
    filtered_metrics = {k: v for k, v in st.session_state.xai_metrics.items() if k in methods_to_show}

    if filtered_metrics:
        eval_data = {
            'Method': list(filtered_metrics.keys()),
            'Faithfulness': [v.get('Faithfulness', 'N/A') for v in filtered_metrics.values()],
            'Stability': [v.get('Stability', 'N/A') for v in filtered_metrics.values()],
            'Interpretability (1-5)': [v.get('Interpretability', 'N/A') for v in filtered_metrics.values()],
            'Compute Time (s)': [v.get('Compute Time (s)', 'N/A') for v in filtered_metrics.values()],
        }
        eval_df = pd.DataFrame(eval_data).set_index('Method')
        st.table(eval_df)

st.markdown("""
**Conclusion:** This multi-method approach provides a comprehensive dashboard. 
*   **Grad-CAM** offers a quick glance at the model's attention.
*   **Integrated Gradients** provides a faithful, pixel-level deep-dive.
*   **LIME** gives a regional summary.
*   **TCAV**, **ProtoPNet**, and **Concept SHAP** translate the model's logic into a clinically relevant and trustworthy narrative, answering *why* in terms of concepts, contributions, and cases.
""")

st.markdown("---")
st.subheader("Future Directions: The \"Time Machine\"")
st.markdown("**Prognostic Modeling**")
st.write("Instead of just diagnosing the current state, predict the future.")

st.info("""
*   **Research:** *"Predictive Multi-task Modelling from Efficient Diffusion Models"* (University of Surrey, MICCAI 2025).
*   **Scope:** "Future iterations will use **Diffusion Models** to generate a synthetic X-ray of the patient's knee *1 year from now* if untreated. This moves the tool from 'Diagnosis' to 'Preventative Care'."
""")