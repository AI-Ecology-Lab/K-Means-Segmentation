import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import tempfile, shutil, io, zipfile, glob
from PIL import Image

# Import image segmentation functions
try:
    from image_segmentation import (
        process_image,
        extract_datetime,
        apply_kmeans,
        preprocess_image,
        merge_clusters,
        generate_stacked_bar_chart,
        generate_color_swatch,
        get_contrasting_colors
    )
except ImportError as e:
    st.error("Error importing image_segmentation: " + str(e))
    st.stop()

# Set page configuration
st.set_page_config(page_title="Image Segmentation with K-Means", page_icon="🖼️", layout="wide")

###############################################################################
# Directory Setup
###############################################################################
@st.cache_resource
def setup_directories():
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'input')
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return temp_dir, input_dir, output_dir

temp_dir, input_dir, output_dir = setup_directories()

###############################################################################
# App Title & Description
###############################################################################
st.title("K-Means Image Segmentation Tool")
st.markdown("""
This app segments images using K-Means clustering. You can:
- Upload images (single, multiple, ZIP file, or a local directory)
- Choose the number of clusters
- View segmented images and cluster statistics
- Merge or rename clusters without widget key collisions
- Generate charts and export your results
""")

###############################################################################
# Sidebar – Settings and File Upload
###############################################################################
with st.sidebar:
    st.header("Settings")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    
    st.header("Upload Options")
    upload_method = st.radio("Upload Method", ["Single Image", "Multiple Images", "ZIP File", "Local Directory"])
    
    if upload_method == "Single Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    elif upload_method == "Multiple Images":
        uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    elif upload_method == "ZIP File":
        uploaded_zip = st.file_uploader("Choose a ZIP file containing images", type=["zip"])
    else:
        input_directory = st.text_input("Enter path to image directory", "")
        use_subdirs = st.checkbox("Include subdirectories", value=True)

###############################################################################
# Helper Functions
###############################################################################
def clear_directory(directory):
    """Delete all files and folders within a directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f"Error deleting {file_path}: {e}")

def save_uploaded_files():
    """Save files from any upload method to the temporary input directory."""
    clear_directory(input_dir)
    file_paths = []
    if upload_method == "Single Image" and uploaded_file is not None:
        file_path = os.path.join(input_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    elif upload_method == "Multiple Images" and uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(input_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
    elif upload_method == "ZIP File" and uploaded_zip is not None:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
            for file in z.namelist():
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('__MACOSX'):
                    z.extract(file, input_dir)
                    file_paths.append(os.path.join(input_dir, file))
    elif upload_method == "Local Directory" and input_directory:
        if not os.path.isdir(input_directory):
            st.error(f"Directory not found: {input_directory}")
            return []
        exts = ['.jpg', '.jpeg', '.png']
        if use_subdirs:
            for ext in exts:
                file_paths.extend(glob.glob(os.path.join(input_directory, f'**/*{ext}'), recursive=True))
                file_paths.extend(glob.glob(os.path.join(input_directory, f'**/*{ext.upper()}'), recursive=True))
        else:
            for ext in exts:
                file_paths.extend(glob.glob(os.path.join(input_directory, f'*{ext}')))
                file_paths.extend(glob.glob(os.path.join(input_directory, f'*{ext.upper()}')))
    return file_paths

def process_images(file_paths):
    """Process uploaded images using K-Means segmentation and return a DataFrame and image list."""
    clear_directory(output_dir)
    all_cluster_data = []
    processed_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_images = len(file_paths)
    st.write(f"Processing {total_images} images...")
    
    # Initialize K-Means with the first image
    status_text.text("Initializing K-Means model...")
    first_image = cv2.imread(file_paths[0])
    if first_image is None:
        st.error(f"Could not read first image: {file_paths[0]}")
        return None, None
    preprocessed_first = preprocess_image(first_image)
    _, _, kmeans_model = apply_kmeans(preprocessed_first, n_clusters)
    
    for i, path in enumerate(file_paths):
        progress_bar.progress((i + 1) / (total_images + 2))
        status_text.text(f"Processing image {i+1}/{total_images}: {os.path.basename(path)}")
        try:
            segmented_image, cluster_data, _, contrasting_image = process_image(path, output_dir, n_clusters, kmeans_model)
            if cluster_data:
                all_cluster_data.append(cluster_data)
                original_image = cv2.imread(path)
                processed_images.append((os.path.basename(path), original_image, segmented_image, contrasting_image))
            else:
                st.warning(f"Processing failed for {os.path.basename(path)}")
        except Exception as e:
            st.error(f"Error processing {os.path.basename(path)}: {e}")
    
    progress_bar.progress(1.0)
    status_text.text("Finalizing data...")
    if all_cluster_data:
        df = pd.DataFrame(all_cluster_data)
        if 'datetime' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception:
                pass
            df = df.sort_values('datetime')
        csv_path = os.path.join(output_dir, 'cluster_data.csv')
        df.to_csv(csv_path, index=False)
        st.success(f"Processing complete! {len(processed_images)}/{total_images} images processed.")
        return df, processed_images
    else:
        st.error("No valid data was extracted from the images.")
        return None, None

def adjust_cluster_percentages(df, source_cluster, target_cluster, offset_percentage):
    """Move a percentage from source cluster to target cluster."""
    adjusted_df = df.copy()
    source_column = f"Cluster_{source_cluster}_Percent"
    target_column = f"Cluster_{target_cluster}_Percent"
    if source_column not in adjusted_df.columns or target_column not in adjusted_df.columns:
        return adjusted_df
    for idx, row in adjusted_df.iterrows():
        source_pct = row[source_column]
        transfer_amount = min(offset_percentage, source_pct)
        adjusted_df.at[idx, source_column] = source_pct - transfer_amount
        adjusted_df.at[idx, target_column] = row[target_column] + transfer_amount
    return adjusted_df

def generate_static_chart(df, x_column='datetime', y_columns=None, title='Cluster Coverage Over Time'):
    """Generate a static stacked bar chart using matplotlib."""
    if y_columns is None:
        y_columns = [col for col in df.columns if col.endswith('_Percent')]
    cluster_names = [col.replace('_Percent', '') for col in y_columns]
    
    # Create a sorted copy of the dataframe based on x values
    plot_df = df.copy()
    
    # If x_column exists, check if we need to sort
    if x_column in plot_df.columns:
        # Check if values are dates or can be converted to numeric
        try:
            # First try to convert to datetime
            plot_df[x_column] = pd.to_datetime(plot_df[x_column])
            plot_df = plot_df.sort_values(x_column)
        except:
            # If not datetime, try to convert to numeric if possible
            try:
                plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors='coerce')
                plot_df = plot_df.sort_values(x_column)
            except:
                # Keep original order if neither works
                pass
    
    fig, ax = plt.subplots(figsize=(12,6))
    x = range(len(plot_df))
    bottom = np.zeros(len(plot_df))
    for i, col in enumerate(y_columns):
        ax.bar(x, plot_df[col], 0.8, bottom=bottom, label=cluster_names[i])
        bottom += plot_df[col].values
    
    ax.set_title(title)
    ax.set_xlabel('Time' if 'datetime' in plot_df.columns else 'Image Number')
    ax.set_ylabel('Percent')
    
    if x_column in plot_df.columns:
        x_labels = plot_df[x_column].astype(str)
        if len(x_labels) > 10:
            indices = np.linspace(0, len(x_labels)-1, 10).astype(int)
            visible_labels = [x_labels.iloc[i] if i in indices else '' for i in range(len(x_labels))]
            ax.set_xticks(x)
            ax.set_xticklabels(visible_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    ax.legend()
    plt.tight_layout()
    return fig

def create_contrast_color_swatch(color, size=(150, 100)):
    """Create an image swatch for a given color."""
    swatch = np.ones((*size, 3), dtype=np.uint8)
    swatch[:] = color
    return swatch

###############################################################################
# New Merge Groups UI – Always Unique Keys!
###############################################################################
def merge_groups_ui(cluster_indices, default_group_count=2):
    """
    Creates UI for merging/renaming clusters.
    Uses unique keys for each widget based on a descriptive prefix and group index.
    """
    # Initialize merge_groups as a dictionary if not present
    if "merge_groups" not in st.session_state:
        st.session_state.merge_groups = {}
        for i in range(default_group_count):
            st.session_state.merge_groups[str(i)] = {"clusters": [], "new_name": f"merged_{i+1}"}
    
    # Let user choose how many groups to create
    group_count = st.number_input("Number of merge groups", min_value=1, max_value=len(cluster_indices),
                                  value=min(len(st.session_state.merge_groups), len(cluster_indices)), key="merge_group_count")
    
    # Adjust the dictionary if group_count has changed
    current_keys = list(st.session_state.merge_groups.keys())
    current_count = len(current_keys)
    
    # Add new groups if needed
    for i in range(current_count, group_count):
        st.session_state.merge_groups[str(i)] = {"clusters": [], "new_name": f"merged_{i+1}"}
    
    # Remove extra groups if needed
    keys_to_remove = []
    for key in st.session_state.merge_groups:
        if int(key) >= group_count:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state.merge_groups[key]
    
    st.markdown("### Define Merge/Rename Groups")
    
    for i in range(group_count):
        key = str(i)
        st.markdown(f"**Group {i+1}**")
        
        # Get existing clusters or empty list
        existing_clusters = st.session_state.merge_groups[key].get("clusters", [])
        # Filter to only valid clusters
        valid_clusters = [c for c in existing_clusters if c in cluster_indices]
        
        clusters = st.multiselect(
            f"Select clusters for Group {i+1}:",
            options=cluster_indices,
            default=valid_clusters,
            key=f"merge_group_{i}_clusters"
        )
        
        new_name = st.text_input(
            f"New name for Group {i+1} cluster:",
            value=st.session_state.merge_groups[key].get("new_name", f"merged_{i+1}"),
            key=f"merge_group_{i}_name"
        )
        
        # Update the session state
        st.session_state.merge_groups[key] = {"clusters": clusters, "new_name": new_name}
    
    return list(st.session_state.merge_groups.values())

###############################################################################
# Main Workflow
###############################################################################
file_paths = save_uploaded_files()
if file_paths:
    df, processed_images = process_images(file_paths)
    if processed_images:
        # For a single image, display it directly.
        if len(processed_images) == 1:
            filename, original_image, segmented_image, contrasting_image = processed_images[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image")
            with col2:
                st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), caption="Segmented Image")
            with col3:
                st.image(cv2.cvtColor(contrasting_image, cv2.COLOR_BGR2RGB), caption="Contrasting Colors")
            st.subheader("Cluster Information")
            for i in range(n_clusters):
                if f"Cluster_{i}_Percent" in df.columns:
                    percent = df.iloc[0][f"Cluster_{i}_Percent"]
                    hsv = df.iloc[0][f"Cluster_{i}_HSV"]
                    st.text(f"Cluster {i}: {percent:.2f}% - HSV: {hsv}")
        else:
            st.success(f"Processed {len(processed_images)} images")
            st.dataframe(df)
            tabs = st.tabs(["Sample Images", "Merge Clusters", "Charts", "Export"])
            
            #############################
            # Tab 1: Sample Images
            #############################
            with tabs[0]:
                st.subheader("Sample Segmented Images")
                sample_size = min(3, len(processed_images))
                sample_indices = np.linspace(0, len(processed_images)-1, sample_size).astype(int)
                samples = [processed_images[i] for i in sample_indices]
                selected_idx = st.selectbox("Select an image to view",
                                            range(len(processed_images)),
                                            format_func=lambda i: processed_images[i][0],
                                            key="sample_image_select")
                filename, original, segmented, contrasting = processed_images[selected_idx]
                st.markdown(f"### {filename}")
                view_mode = st.radio("Display Mode", ["Side by Side", "Original vs Contrasting", "Clusters with Colors"],
                                     key="display_mode_select")
                if view_mode == "Side by Side":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original")
                    with col2:
                        st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), caption="Segmented")
                elif view_mode == "Original vs Contrasting":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original")
                    with col2:
                        st.image(cv2.cvtColor(contrasting, cv2.COLOR_BGR2RGB), caption="Contrasting")
                else:
                    st.image(cv2.cvtColor(contrasting, cv2.COLOR_BGR2RGB), caption="Clusters with Contrasting Colors")
            
            #############################
            # Tab 2: Merge Clusters
            #############################
            with tabs[1]:
                st.subheader("Merge or Rename Clusters (CSV Update Only)")
                percent_columns = [col for col in df.columns if col.endswith('_Percent')]
                cluster_indices = [col.split('_')[1] for col in percent_columns]
                merge_groups = merge_groups_ui(cluster_indices)
                
                # Validate merge groups
                merge_enabled = True
                error_message = ""
                all_selected = []
                for group in merge_groups:
                    if len(group["clusters"]) < 1:
                        merge_enabled = False
                        error_message = "Each group must include at least 1 cluster."
                        break
                    all_selected.extend(group["clusters"])
                if len(all_selected) != len(set(all_selected)):
                    merge_enabled = False
                    error_message = "A cluster cannot be assigned to more than one group."
                if not merge_enabled:
                    st.error(error_message)
                else:
                    if st.button("Merge/Rename Clusters", key="merge_clusters_btn"):
                        df_merged = df.copy()
                        for group in merge_groups:
                            clusters_to_merge = group["clusters"]
                            new_name = group["new_name"]
                            merged_percent_col = f"Cluster_{new_name}_Percent"
                            merged_hsv_col = f"Cluster_{new_name}_HSV"
                            merged_percent = df_merged[[f"Cluster_{c}_Percent" for c in clusters_to_merge]].sum(axis=1)
                            df_merged[merged_percent_col] = merged_percent
                            df_merged[merged_hsv_col] = df_merged[f"Cluster_{clusters_to_merge[0]}_HSV"]
                            for c in clusters_to_merge:
                                if f"Cluster_{c}_Percent" in df_merged.columns:
                                    del df_merged[f"Cluster_{c}_Percent"]
                                if f"Cluster_{c}_HSV" in df_merged.columns:
                                    del df_merged[f"Cluster_{c}_HSV"]
                        st.session_state.df_merged = df_merged
                        st.success("Clusters merged/renamed successfully!")
                        st.dataframe(df_merged)
            
            #############################
            # Tab 3: Charts
            #############################
            with tabs[2]:
                st.subheader("Generate Charts")
                use_merged = st.checkbox("Use merged clusters", value=False, key="use_merged_chart")
                use_adjusted = st.checkbox("Use adjusted percentages", value=False, key="use_adjusted_chart",
                                           disabled='adjusted_df' not in st.session_state)
                add_trendline = st.checkbox("Add trend lines", value=False, key="add_trendline_chart")
                chart_type = st.radio("Chart Type", ["Interactive", "Static"], key="chart_type_chart")
                if st.button("Generate Chart", key="generate_chart_btn"):
                    plot_df = df
                    if use_adjusted and 'adjusted_df' in st.session_state:
                        plot_df = st.session_state.adjusted_df
                    elif use_merged and 'df_merged' in st.session_state:
                        plot_df = st.session_state.df_merged
                    y_columns = [col for col in plot_df.columns if col.endswith('_Percent')]
                    if chart_type == "Interactive":
                        fig = generate_stacked_bar_chart(plot_df, 'datetime', y_columns, add_trendline=add_trendline)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = generate_static_chart(plot_df, 'datetime', y_columns)
                        st.pyplot(fig)
            
            #############################
            # Tab 4: Export
            #############################
            with tabs[3]:
                st.subheader("Export Results")
                use_merged_export = st.checkbox("Use merged data for export", value=False, key="use_merged_export")
                use_adjusted_export = st.checkbox("Use adjusted percentages for export", value=False, key="use_adjusted_export",
                                                  disabled='adjusted_df' not in st.session_state)
                add_trendline_export = st.checkbox("Include trend lines in exported chart", value=False, key="add_trendline_export")
                chart_filename = st.text_input("Chart filename", value="cluster_coverage_chart.html", key="chart_filename_input")
                if st.button("Export Data and Chart", key="export_button"):
                    export_df = df
                    if use_adjusted_export and 'adjusted_df' in st.session_state:
                        export_df = st.session_state.adjusted_df
                    elif use_merged_export and 'df_merged' in st.session_state:
                        export_df = st.session_state.df_merged
                    export_csv_path = os.path.join(output_dir, 'final_cluster_data.csv')
                    export_df.to_csv(export_csv_path, index=False)
                    y_columns = [col for col in export_df.columns if col.endswith('_Percent')]
                    fig = generate_stacked_bar_chart(export_df, 'datetime', y_columns, add_trendline=add_trendline_export)
                    chart_path = os.path.join(output_dir, chart_filename)
                    fig.write_html(chart_path)
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(export_csv_path, os.path.basename(export_csv_path))
                        zf.write(chart_path, os.path.basename(chart_path))
                        for fname in os.listdir(output_dir):
                            if fname.startswith("segmented_") and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                fpath = os.path.join(output_dir, fname)
                                zf.write(fpath, fname)
                    zip_buffer.seek(0)
                    st.download_button("Download All Results", data=zip_buffer, file_name="segmentation_results.zip", mime="application/zip")
                    st.success("Export complete!")
    else:
        st.error("Image processing did not yield any results.")
else:
    st.info("Please upload images or provide a directory to begin processing.")

###############################################################################
# Footer
###############################################################################
st.markdown("---")
st.markdown("K-Means Image Segmentation Tool - oceancv.org")

###############################################################################
# Color Legend
###############################################################################
with st.expander("Show Cluster Color Legend", expanded=False):
    st.write("These contrasting colors are used to visualize different clusters in the segmented images:")
    
    # Get the colors from image_segmentation module
    from image_segmentation import CONTRASTING_COLORS
    
    # Create color swatches
    color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Purple", "Brown", "Teal", "Dark Red"]
    
    # Define how many colors to display per row
    colors_per_row = 5
    for i in range(0, len(CONTRASTING_COLORS), colors_per_row):
        cols = st.columns(colors_per_row)
        for j in range(colors_per_row):
            idx = i + j
            if idx < len(CONTRASTING_COLORS):
                # Get color and convert BGR to RGB for display
                bgr_color = CONTRASTING_COLORS[idx]
                rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])  # Convert BGR to RGB
                
                # Create HTML color code
                html_color = f'rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})'
                
                # Display in column with colored box
                with cols[j]:
                    st.markdown(f"""
                    <div style="
                        background-color: {html_color}; 
                        width: 100%; 
                        height: 50px; 
                        border-radius: 5px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-bottom: 5px;
                    ">
                    </div>
                    <div style="text-align: center;">
                        <b>Cluster {idx+1}</b><br>
                        {color_names[idx]}
                    </div>
                    """, unsafe_allow_html=True)