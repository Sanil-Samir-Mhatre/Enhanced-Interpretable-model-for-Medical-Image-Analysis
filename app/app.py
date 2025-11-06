import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import streamlit as st 
import tensorflow as tf
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import warnings


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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


def get_integrated_gradients(baseline, image, target_class_idx, m_steps=50):
    # 1. Generate interpolated images
    interpolated_images = tf.convert_to_tensor(
        [
            baseline + (float(i) / m_steps) * (image - baseline)
            for i in range(m_steps + 1)
        ]
    )

    # 2. Get gradients of predictions with respect to interpolated images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        # The model expects a batch, so we pass the interpolated images
        preds = model(interpolated_images)
        # We are interested in the gradients for the predicted class
        top_class_preds = preds[:, target_class_idx]

    grads = tape.gradient(top_class_preds, interpolated_images)

    # 3. Approximate the integral using the trapezoidal rule
    # This is equivalent to averaging the gradients and scaling by the difference
    # between the input and the baseline.
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_gradients = (image - baseline) * avg_grads
    return integrated_gradients


def visualize_attributions(attributions):
    # Sum the attributions across color channels and take the absolute value.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    # Normalize the mask to the [0, 1] range.
    attribution_mask /= tf.reduce_max(attribution_mask)
    return attribution_mask

icon_path = os.path.join(SCRIPT_DIR, "img", "mdc.png")
icon = Image.open(icon_path)
st.set_page_config(
    page_title="Severity Analysis of Osteoarthritis in the Knee",
    page_icon=icon,
)
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
model_path = os.path.join(SCRIPT_DIR, "..", "src", "models", "model_Xception_ft.hdf5")
model = tf.keras.models.load_model(model_path)
target_size = (224, 224)

# Grad-CAM
grad_model = tf.keras.models.clone_model(model)
grad_model.set_weights(model.get_weights())
grad_model.layers[-1].activation = None
grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# Sidebar
with st.sidebar:
    st.image(icon)
    st.subheader("Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")


# Body
st.header("Severity Analysis of Osteoarthritis in the Knee")

st.markdown("""
This application analyzes knee X-ray images to predict the severity of osteoarthritis based on the Kellgren-Lawrence (KL) grading system. 
The underlying deep learning model is an *Xception* model, which achieves a *Balanced Accuracy of 67%* on the test dataset.
""")

with st.expander("See other model performances"):
    st.markdown("""
    | Model                           | Balanced Accuracy |
    | ------------------------------- | ----------------- |
    | Ensemble (f1-score weighted)    | 68.69%            |
    | *Xception fine-tuning*        | *67%*           |
    | ResNet50 fine-tuning            | 65%               |
    | Inception-ResNet-v2 fine-tuning | 64%               |
    """)

col1, col2 = st.columns(2)

# Initialize session state
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.y_pred = None
    st.session_state.img_array = None

if uploaded_file is not None:
    # Load the image once, so it's available for all parts of the app
    # Load original image for high-quality display
    original_image = Image.open(uploaded_file).convert("RGB")
    # Load and resize image for model prediction
    img_for_prediction = original_image.resize(target_size)
    img_for_display = tf.keras.preprocessing.image.img_to_array(img_for_prediction)

    with col1:
        st.subheader("Input")
        st.image(uploaded_file, width='stretch') # Keeping this as is, as it's widely compatible. The warning is for future versions.

        if st.button(
            "Predict Osteoarthritis in the Knee"
        ):
            img_array = np.expand_dims(img_for_display, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(
                img_array
            )

            with st.spinner("Wait for it..."):
                y_pred = model.predict(img_array)
                st.session_state.prediction_made = True
                st.session_state.y_pred = 100 * y_pred[0]
                st.session_state.img_array = img_array
            
            y_pred = 100 * y_pred[0]

            probability = np.amax(y_pred)
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(class_names[np.amax(number)])

            st.subheader("Prediction")

            st.metric(
                label="Severity Grade:",
                value=f"{grade} - {probability:.2f}%",
            )
            st.caption(
                "This percentage represents the model's confidence in its prediction. The model assigns a probability to each of the 5 KL grades, and this is the highest one."
            )
            
            # Integrated Gradients Explanation
            st.subheader("Integrated Gradients Explanation")
            st.info(
                '*Integrated Gradients answers:* "Which pixels were most influential in the final prediction?" It provides a fine-grained attribution map by integrating gradients along a path from a baseline (black image) to the input image.'
            )
            with st.spinner("Calculating Integrated Gradients..."):
                # The model expects preprocessed input
                ig_img_array = tf.keras.applications.xception.preprocess_input(np.expand_dims(img_for_display, axis=0))[0]
                baseline = np.zeros(target_size + (3,)).astype(np.float32)
                
                pred_class_idx = np.argmax(st.session_state.y_pred)
                attributions = get_integrated_gradients(baseline=baseline, image=ig_img_array, target_class_idx=pred_class_idx)
                attribution_mask = visualize_attributions(attributions)

                # Display the original image and the IG mask
                st.image(attribution_mask.numpy(), clamp=True, width='stretch', caption="Integrated Gradients highlights influential pixels for the prediction.")

    if st.session_state.get("prediction_made"):
        with col2:
            st.subheader("Explanation using GradCam Heatmaps")
            st.info(
                '*Grad-CAM answers:* "Where in the image did the final convolutional neurons fire the most intensely for this prediction?" It\'s a direct look at the model\'s internal state—a heat map of attention.'
            )
            heatmap = make_gradcam_heatmap(grad_model, st.session_state.img_array)            
            gradcam_image = save_and_display_gradcam(original_image, heatmap)
            st.image(gradcam_image, width='stretch')

            st.subheader("Analysis")
            y_pred_for_chart = st.session_state.y_pred
            kl_grade = np.argmax(y_pred_for_chart)
            st.write(f"The predicted KL Grade is: *{kl_grade}*")
            st.caption("KL Grade ranges from 0 (Healthy) to 4 (Severe).")
            
            # LIME Explanation
            st.subheader("Superpixel-based Explanation")
            st.info(
                '*This method answers:* "Which high-level features (superpixels), if removed, would change the prediction the most?" It provides a feature-importance ranking, which is a different and often more human-intuitive form of explanation.'
            )

            # Add a placeholder for the progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            def predict_fn_for_lime(images):
                # LIME generates images with float64, need to convert for preprocessing
                images_float32 = images.astype('float32')
                img_processed = tf.keras.applications.xception.preprocess_input(images_float32)
                return model.predict(img_processed)

            status_text.text("Generating superpixels and perturbations...")
            progress_bar.progress(25)

            # 1. Create an image explainer with a fixed random state for reproducibility
            explainer = lime_image.LimeImageExplainer(random_state=42)

            # Create a segmentation function with our desired settings
            segmentation_fn = lambda x: slic(x, n_segments=100, compactness=10, sigma=1, start_label=1)

            # 2. Generate an explanation with a better segmentation algorithm (slic)
            explanation = explainer.explain_instance(
                np.array(img_for_prediction), 
                predict_fn_for_lime, 
                top_labels=1, 
                hide_color=0, 
                num_samples=1000, # Number of samples to generate for explanation,
                segmentation_fn=segmentation_fn
            )
            
            status_text.text("Generating explanation image...")
            progress_bar.progress(75)
            # 3. Get the image and mask, showing more features and only positive contributions
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
            )
            lime_img = mark_boundaries(temp / 2 + 0.5, mask)
            st.image(lime_img, clamp=True, width='stretch', caption="The explanation highlights regions positively contributing to the prediction.")
            
            # Clean up
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

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
The integration of three distinct explainability techniques—Grad-CAM, Integrated Gradients, and Superpixel Analysis (LIME)—is crucial for building a transparent and trustworthy diagnostic aid. By implementing these XAI methods, we move beyond "black box" predictions and provide insights into the model's decision-making process.

For doctors, this transparency is invaluable. It allows them to:
*   *Verify Predictions:* Clinicians can cross-reference the model's highlighted regions (via heatmaps and masks) with their own expert assessment of the X-ray, confirming if the model is focusing on clinically relevant features like joint space narrowing or osteophytes.
*   *Build Confidence:* Seeing why a prediction was made helps build trust in the tool. If the explanations are nonsensical, it signals that the prediction may be unreliable.
*   *Gain New Insights:* The model might identify subtle patterns that are not immediately obvious, offering a new perspective for the clinician to consider.

Ultimately, providing these explanations transforms the application from a simple prediction tool into a collaborative partner, augmenting a doctor's expertise rather than just offering an opinion.
""")
