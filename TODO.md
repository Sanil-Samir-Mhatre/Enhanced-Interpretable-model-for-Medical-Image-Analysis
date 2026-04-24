# TODO: Implement GradCAM + LIME Hybrid Model (after 1st Hybrid)

## Plan Summary
Add new hybrid model combining GradCAM (heatmap) and LIME (sparse superpixel mask) via element-wise multiplication for sparse localized heatmap. Insert as "5. Hybrid XAI #2" after existing Hybrid #1 (GradCAM+IG) in app/app.py.

## Steps to Complete
### 1. [x] Add new functions in app/app.py
- `generate_gradcam_lime_hybrid(gradcam_heatmap, lime_mask)`
- `@st.cache_data def get_gradcam_lime_hybrid_explanation(...)`

### 2. [x] Insert new UI section "5. Hybrid XAI #2 (Grad-CAM + LIME)"
- After existing Hybrid #1 section.
- Include spinner, image display, download, metrics, anatomical expander.

### 3. [x] Update evaluation table
- Add 'Hybrid2 (GradCAM+LIME)' to filtered_metrics.

### 4. [x] Test
- Run `streamlit run app/app.py`
- Upload image, verify new section #5 works, metrics shown.

### 5. [x] Mark complete & attempt_completion

