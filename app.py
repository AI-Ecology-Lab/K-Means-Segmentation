import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import shutil
from PIL import Image
import io
import zipfile
import glob
try:
    from image_segmentation import (
        process_image, 
        extract_datetime, 
        apply_kmeans, 
        preprocess_image,
        merge_clusters, 
        generate_stacked_bar_chart,
        generate_color_swatch,
        get_contrasting_colors  # Add this import to fix the error
    )
except ImportError as e:
    st.error("Error importing image_segmentation: " + str(e))
    st.stop()

# Set page config
st.set_page_config(
    page_title="Image Segmentation with K-Means",
    page_icon="🖼️",
    layout="wide",
)

# Set up directories
@st.cache_resource
def setup_directories():
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'input')
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return temp_dir, input_dir, output_dir

temp_dir, input_dir, output_dir = setup_directories()

# Title and description
st.title("K-Means Image Segmentation Tool")
st.markdown("""
This app allows you to segment images using K-Means clustering. You can:
- Upload single images or multiple images
- Adjust number of clusters
- View segmented images
- Analyze cluster distributions
- Merge similar clusters
- Generate time-series charts with trend lines
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    
    st.header("Upload")
    upload_method = st.radio("Upload Method", ["Single Image", "Multiple Images", "ZIP File", "Local Directory"])
    
    if upload_method == "Single Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    elif upload_method == "Multiple Images":
        uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    elif upload_method == "ZIP File":
        uploaded_zip = st.file_uploader("Choose ZIP file with images", type=["zip"])
    else:  # Local Directory
        input_directory = st.text_input("Enter path to image directory", "")
        use_subdirs = st.checkbox("Include subdirectories", value=True)

# Function to clear directories
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f"Error deleting {file_path}: {e}")

# Function to display a single segmented image
def display_segmented_image(original_image, segmented_image, cluster_info):
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image")
    with col2:
        st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), caption="Segmented Image")
    
    # Display cluster information
    st.write("Cluster Information:")
    for info in cluster_info:
        st.text(info)

# Save uploaded files to the input directory
def save_uploaded_files():
    clear_directory(input_dir)
    
    if upload_method == "Single Image" and uploaded_file is not None:
        # Save the single uploaded file
        file_path = os.path.join(input_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return [file_path]
    
    elif upload_method == "Multiple Images" and uploaded_files:
        # Save multiple uploaded files
        file_paths = []
        for file in uploaded_files:
            file_path = os.path.join(input_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        return file_paths
    
    elif upload_method == "ZIP File" and uploaded_zip is not None:
        # Extract and save files from ZIP
        file_paths = []
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
            for file in z.namelist():
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('__MACOSX'):
                    z.extract(file, input_dir)
                    file_path = os.path.join(input_dir, file)
                    file_paths.append(file_path)
        return file_paths
    
    elif upload_method == "Local Directory" and input_directory:
        # Check if the directory exists
        if not os.path.isdir(input_directory):
            st.error(f"Directory not found: {input_directory}")
            return []
        
        # Get all image files in the directory
        file_paths = []
        if use_subdirs:
            # Include subdirectories
            for ext in ['.jpg', '.jpeg', '.png']:
                file_paths.extend(glob.glob(os.path.join(input_directory, f'**/*{ext}'), recursive=True))
                file_paths.extend(glob.glob(os.path.join(input_directory, f'**/*{ext.upper()}'), recursive=True))
        else:
            # Only the main directory
            for ext in ['.jpg', '.jpeg', '.png']:
                file_paths.extend(glob.glob(os.path.join(input_directory, f'*{ext}')))
                file_paths.extend(glob.glob(os.path.join(input_directory, f'*{ext.upper()}')))
        
        if not file_paths:
            st.error(f"No image files found in {input_directory}")
            return []
        
        # For local directory, we'll use the files directly from their location
        # No need to copy them to the temp input directory
        return file_paths
    
    return []

# Process images and get results
def process_images(file_paths):
    clear_directory(output_dir)
    
    all_cluster_data = []
    processed_images = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Show count and details
    total_images = len(file_paths)
    st.write(f"Processing {total_images} images...")
    
    # Initialize KMeans model with the first image
    status_text.text("Step 1/3: Initializing K-Means model with first image...")
    first_image = cv2.imread(file_paths[0])
    if first_image is None:
        st.error(f"Could not read first image: {file_paths[0]}. Please check the file.")
        return None, None
        
    preprocessed_first = preprocess_image(first_image)
    _, _, kmeans_model = apply_kmeans(preprocessed_first, n_clusters)
    
    # Create expander for detailed logs
    with st.expander("Processing Details", expanded=False):
        log_container = st.container()
        log_container.write("Processing Log:")
        log_text = ""
    
    # Process each image
    status_text.text("Step 2/3: Processing individual images...")
    for i, path in enumerate(file_paths):
        # Update progress bar and status
        progress = (i + 1) / (total_images + 2)  # +2 for initialization and finalization steps
        progress_bar.progress(progress)
        status_text.text(f"Processing image {i+1}/{total_images}: {os.path.basename(path)}")
        
        # Process image
        log_text += f"Processing: {os.path.basename(path)} - "
        try:
            segmented_image, cluster_data, _, contrasting_image = process_image(
                path, output_dir, n_clusters, kmeans_model
            )
            
            if cluster_data:
                all_cluster_data.append(cluster_data)
                original_image = cv2.imread(path)
                processed_images.append((os.path.basename(path), original_image, segmented_image, contrasting_image))
                log_text += f"✓ Success - {len(cluster_data.keys())-2} clusters extracted\n"
            else:
                log_text += f"✗ Failed to process\n"
        except Exception as e:
            log_text += f"✗ Error: {str(e)}\n"
        
        # Update log display
        log_container.text(log_text)
    
    # Create DataFrame from cluster data
    status_text.text("Step 3/3: Creating data frame and saving results...")
    if all_cluster_data:
        df = pd.DataFrame(all_cluster_data)
        
        # Sort by datetime/identifier for consistent ordering
        if 'datetime' in df.columns:
            # Check if all values are numeric
            if all(isinstance(x, (int, float)) for x in df['datetime'] if not pd.isna(x)):
                df['datetime'] = pd.to_numeric(df['datetime'], errors='coerce')
            # Sort the DataFrame
            df = df.sort_values('datetime')
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'cluster_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text(f"Processing complete! Successfully processed {len(processed_images)}/{total_images} images.")
        
        return df, processed_images
    else:
        progress_bar.progress(1.0)
        status_text.text("Processing failed. No valid data was extracted from the images.")
        return None, None

# Add this function to adjust cluster percentages
def adjust_cluster_percentages(df, source_cluster, target_cluster, offset_percentage):
    """
    Adjust cluster percentages by moving a percentage from one cluster to another.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cluster data
    source_cluster : str
        Cluster to take percentage from
    target_cluster : str
        Cluster to add percentage to
    offset_percentage : float
        Percentage to move from source to target
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with adjusted percentages
    """
    adjusted_df = df.copy()
    source_column = f"Cluster_{source_cluster}_Percent"
    target_column = f"Cluster_{target_cluster}_Percent"
    
    if source_column not in adjusted_df.columns or target_column not in adjusted_df.columns:
        return adjusted_df
    
    # Calculate the actual amount to transfer for each row
    for idx, row in adjusted_df.iterrows():
        # Get source percentage
        source_pct = row[source_column]
        
        # Calculate amount to transfer (limited by available percentage)
        transfer_amount = min(offset_percentage, source_pct)
        
        # Adjust source and target percentages
        adjusted_df.at[idx, source_column] = source_pct - transfer_amount
        adjusted_df.at[idx, target_column] = row[target_column] + transfer_amount
    
    return adjusted_df

# Add this function to generate static charts
def generate_static_chart(df, x_column='datetime', y_columns=None, title='Percent Cover of Clusters Over Time'):
    """Generate a static matplotlib stacked bar chart"""
    import matplotlib.pyplot as plt
    
    # If y_columns not specified, use all percent columns
    if y_columns is None:
        y_columns = [col for col in df.columns if col.endswith('_Percent')]
    
    # Get cluster names for legend
    cluster_names = [col.replace('_Percent', '') for col in y_columns]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get x-axis values
    x = range(len(df))
    width = 0.8
    
    # Initialize bottom for stacked bars
    bottom = np.zeros(len(df))
    
    # Plot each cluster as a bar segment
    for i, col in enumerate(y_columns):
        ax.bar(x, df[col], width, bottom=bottom, label=cluster_names[i])
        bottom += df[col].values
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Percent Cover')
    
    # Set x-ticks to show datetime or identifiers
    if x_column in df.columns:
        x_labels = df[x_column].astype(str)
        if len(x_labels) > 10:
            # If too many labels, show only some of them
            indices = np.linspace(0, len(x_labels)-1, 10).astype(int)
            visible_labels = [x_labels.iloc[i] if i in indices else '' for i in range(len(x_labels))]
            ax.set_xticks(x)
            ax.set_xticklabels(visible_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Add this function to create a contrasting color swatch
def create_contrast_color_swatch(color, size=(150, 100)):
    """Create a swatch for a contrasting color"""
    # Create a solid color image
    swatch = np.ones((*size, 3), dtype=np.uint8)
    swatch[:] = color
    return swatch

# Main processing workflow
if ((upload_method == "Single Image" and uploaded_file is not None) or
    (upload_method == "Multiple Images" and uploaded_files) or
    (upload_method == "ZIP File" and uploaded_zip is not None) or
    (upload_method == "Local Directory" and input_directory)):
    
    with st.spinner("Preparing files..."):
        file_paths = save_uploaded_files()
    
    if file_paths:
        if len(file_paths) == 1 and upload_method == "Single Image":
            # Process single image 
            st.info("Processing single image...")
            df, processed_images = process_images(file_paths)
            
            if processed_images:
                filename, original_image, segmented_image, contrasting_image = processed_images[0]
                
                # Extract cluster info
                cluster_info = []
                for i in range(n_clusters):
                    if f'Cluster_{i}_Percent' in df.iloc[0]:
                        percent = df.iloc[0][f'Cluster_{i}_Percent']
                        hsv = df.iloc[0][f'Cluster_{i}_HSV']
                        cluster_info.append(f"Cluster {i}: {percent:.2f}% - HSV: {hsv}")
                
                # Display the segmented image with both versions
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image")
                with col2:
                    st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), caption="Segmented Image")
                with col3:
                    st.image(cv2.cvtColor(contrasting_image, cv2.COLOR_BGR2RGB), caption="Contrasting Colors")
                
                # Display cluster information with color swatches
                st.subheader("Cluster Information")
                
                clusters_per_row = 3
                cluster_rows = [cluster_info[i:i+clusters_per_row] for i in range(0, len(cluster_info), clusters_per_row)]
                
                for row in cluster_rows:
                    cols = st.columns(clusters_per_row)
                    for i, (info, col) in enumerate(zip(row, cols)):
                        cluster_idx = i + (cluster_rows.index(row) * clusters_per_row)
                        hsv_value = df.iloc[0][f'Cluster_{cluster_idx}_HSV']
                        
                        # Create color swatch
                        try:
                            swatch = generate_color_swatch(hsv_value)
                            with col:
                                st.image(swatch, caption=f"Cluster {cluster_idx}")
                                st.text(f"HSV: {hsv_value}")
                                st.text(f"Coverage: {df.iloc[0][f'Cluster_{cluster_idx}_Percent']:.2f}%")
                        except Exception as e:
                            with col:
                                st.error(f"Could not display swatch: {str(e)}")
        else:
            # Process multiple images - no need for spinner as we have progress bar now
            df, processed_images = process_images(file_paths)
            
            if df is not None and processed_images:
                st.success(f"Processed {len(processed_images)} images")
                
                # Create a data metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Images Processed", f"{len(processed_images)}/{len(file_paths)}")
                with col2:
                    st.metric("Clusters Identified", n_clusters)
                with col3:
                    st.metric("Success Rate", f"{len(processed_images)/len(file_paths)*100:.1f}%")
                with col4:
                    total_pixels = sum(df[f'Cluster_{i}_Percent'].sum() for i in range(n_clusters) if f'Cluster_{i}_Percent' in df.columns)
                    st.metric("Total Pixels Analyzed", f"{total_pixels:.0f}%")
                
                # Display path info if using local directory
                if upload_method == "Local Directory":
                    st.info(f"Images processed from: {input_directory}")
                
                # Display identifier type
                if 'datetime' in df.columns:
                    if all(isinstance(x, (int, float)) for x in df['datetime'] if not pd.isna(x)):
                        st.info("Images have numeric identifiers (e.g., 0.jpg, 1.jpg)")
                    elif all(isinstance(x, datetime) for x in df['datetime'] if not pd.isna(x)):
                        st.info("Images have datetime identifiers")
                    else:
                        st.info("Images have mixed or string identifiers")
                
                # Display the data and create tabs for processing
                st.subheader("Cluster Data")
                st.dataframe(df)
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(["Sample Images", "Merge Clusters", "Charts", "Export"])
                
                with tab1:
                    st.subheader("Sample Segmented Images")
                    # Select a few sample images to display
                    sample_size = min(3, len(processed_images))
                    sample_indices = np.linspace(0, len(processed_images)-1, sample_size).astype(int)
                    samples = [processed_images[i] for i in sample_indices]
                    
                    # Add image selector
                    if len(processed_images) > 0:
                        selected_image_idx = st.selectbox(
                            "Select image to view:",
                            range(len(processed_images)),
                            format_func=lambda i: processed_images[i][0]
                        )
                        
                        if selected_image_idx is not None:
                            selected_filename, selected_original, selected_segmented, selected_contrasting = processed_images[selected_image_idx]
                            st.markdown(f"### Selected Image: {selected_filename}")
                            
                            display_mode = st.radio(
                                "Display Mode:",
                                ["Side by Side", "Original vs Contrasting", "Clusters with Colors"]
                            )
                            
                            if display_mode == "Side by Side":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(cv2.cvtColor(selected_original, cv2.COLOR_BGR2RGB), caption="Original")
                                with col2:
                                    st.image(cv2.cvtColor(selected_segmented, cv2.COLOR_BGR2RGB), caption="Segmented")
                            
                            elif display_mode == "Original vs Contrasting":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(cv2.cvtColor(selected_original, cv2.COLOR_BGR2RGB), caption="Original")
                                with col2:
                                    st.image(cv2.cvtColor(selected_contrasting, cv2.COLOR_BGR2RGB), caption="Contrasting Colors")
                            
                            else:  # "Clusters with Colors"
                                # Show contrasting image
                                st.image(cv2.cvtColor(selected_contrasting, cv2.COLOR_BGR2RGB), caption="Clusters with Contrasting Colors")
                                
                                # Get cluster data for this image
                                if df is not None and 'filename' in df.columns:
                                    image_data = df[df['filename'] == selected_filename]
                                    
                                    if not image_data.empty:
                                        st.markdown("### Cluster Colors and Values")
                                        
                                        # Calculate how many clusters to display per row
                                        clusters_per_row = 4
                                        percent_columns = [col for col in image_data.columns if col.endswith('_Percent')]
                                        num_clusters = len(percent_columns)
                                        
                                        # Create rows of clusters
                                        for start_idx in range(0, num_clusters, clusters_per_row):
                                            end_idx = min(start_idx + clusters_per_row, num_clusters)
                                            cols = st.columns(clusters_per_row)
                                            
                                            for i, col_idx in enumerate(range(start_idx, end_idx)):
                                                cluster_name = percent_columns[col_idx].split('_')[1]
                                                hsv_key = f"Cluster_{cluster_name}_HSV"
                                                percent_key = percent_columns[col_idx]
                                                
                                                if hsv_key in image_data.columns:
                                                    hsv_value = image_data.iloc[0][hsv_key]
                                                    percent_value = image_data.iloc[0][percent_key]
                                                    
                                                    # Generate color swatch
                                                    try:
                                                        swatch = generate_color_swatch(hsv_value)
                                                        with cols[i]:
                                                            st.image(swatch, caption=f"Cluster {cluster_name}")
                                                            st.text(f"HSV: {hsv_value}")
                                                            st.text(f"Coverage: {percent_value:.2f}%")
                                                    except Exception as e:
                                                        with cols[i]:
                                                            st.error(f"Could not display: {str(e)}")
                    
                    # Show the original sample images
                    st.markdown("### Sample Images")
                    for filename, original, segmented, contrasting in samples:
                        st.markdown(f"**File: {filename}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original")
                        with col2:
                            st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), caption="Segmented")
                        with col3:
                            st.image(cv2.cvtColor(contrasting, cv2.COLOR_BGR2RGB), caption="Contrasting")
                
                    # Add a new section for cluster color swatches
                    st.markdown("### Cluster Color Swatches")
                    
                    if df is not None:
                        # Get a sample row (first row) for cluster information
                        sample_row = df.iloc[0]
                        
                        # Determine how many clusters we have
                        percent_columns = [col for col in sample_row.index if col.endswith('_Percent')]
                        num_clusters = len(percent_columns)
                        
                        # Create two tabs for original and contrasting colors
                        swatch_tab1, swatch_tab2 = st.tabs(["Original Colors", "Contrasting Colors"])
                        
                        with swatch_tab1:
                            # Display original cluster colors
                            cols = st.columns(min(4, num_clusters))
                            for i in range(num_clusters):
                                cluster_name = percent_columns[i].split('_')[1]
                                hsv_key = f"Cluster_{cluster_name}_HSV"
                                
                                if hsv_key in sample_row:
                                    hsv_value = sample_row[hsv_key]
                                    try:
                                        cluster_color = generate_color_swatch(hsv_value, size=(150, 100))
                                        with cols[i % len(cols)]:
                                            st.image(cluster_color, caption=f"Cluster {cluster_name}")
                                            st.text(f"HSV: {hsv_value}")
                                    except Exception as e:
                                        with cols[i % len(cols)]:
                                            st.error(f"Error with Cluster {cluster_name}: {str(e)}")
                        
                        with swatch_tab2:
                            # Display contrasting colors
                            contrasting_colors = get_contrasting_colors(num_clusters)
                            cols = st.columns(min(4, num_clusters))
                            
                            for i in range(num_clusters):
                                cluster_name = percent_columns[i].split('_')[1]
                                
                                # Get the predefined contrasting color for this cluster
                                color = contrasting_colors[i]
                                
                                # Create a swatch with the contrasting color
                                swatch = create_contrast_color_swatch(color)
                                
                                # Display RGB values (convert BGR to RGB)
                                rgb_color = (color[2], color[1], color[0])
                                
                                with cols[i % len(cols)]:
                                    st.image(cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB), caption=f"Cluster {cluster_name}")
                                    st.text(f"RGB: {rgb_color}")

with tab2:
    st.subheader("Merge or Rename Clusters")

    if df is not None:
        # Extract available cluster IDs from columns (e.g. "Cluster_1_Percent")
        percent_columns = [col for col in df.columns if col.endswith('_Percent')]
        cluster_indices = [col.split('_')[1] for col in percent_columns]

        st.markdown("### Step 1: Set Merge/Rename Parameters")
        st.write("Define the desired final count of clusters. Based on that, you can define groups where each group will be merged (or renamed if only one cluster is selected).")
        
        desired_final_count = st.number_input(
            "Enter desired final count of clusters:",
            min_value=1,
            max_value=len(cluster_indices),
            value=len(cluster_indices),
            step=1
        )

        # Calculate how many groups are needed to achieve the reduction.
        merge_group_count = len(cluster_indices) - desired_final_count
        st.write(f"To reduce from {len(cluster_indices)} to {desired_final_count} clusters, you can create up to **{merge_group_count} merge/rename group(s)**.")

        merge_groups = {}
        if merge_group_count > 0:
            st.markdown("### Step 2: Define Merge/Rename Groups")
            st.write("For each group, select one or more clusters to merge (or simply rename) and provide a new name for the resulting cluster.")
            for i in range(merge_group_count):
                st.markdown(f"**Group {i+1}**")
                selected_clusters = st.multiselect(
                    f"Select clusters for group {i+1}:",
                    options=cluster_indices,
                    key=f"group_{i}_clusters"
                )
                new_name = st.text_input(
                    f"New name for group {i+1} cluster:",
                    value=f"merged_{i+1}",
                    key=f"group_{i}_name"
                )
                merge_groups[i] = {"clusters": selected_clusters, "new_name": new_name}

        st.markdown("### Step 3: Execute Merge/Rename")
        merge_enabled = True
        error_message = ""

        # Validate each group: each must have at least 1 cluster selected.
        if merge_group_count > 0:
            for i, group in merge_groups.items():
                if len(group["clusters"]) < 1:
                    merge_enabled = False
                    error_message = f"Group {i+1} must include at least 1 cluster."
                    break

            # Ensure no cluster appears in more than one group.
            all_selected = []
            for group in merge_groups.values():
                all_selected.extend(group["clusters"])
            if len(all_selected) != len(set(all_selected)):
                merge_enabled = False
                error_message = "A cluster cannot be assigned to more than one group."

        if not merge_enabled:
            st.error(error_message)

        if merge_enabled and st.button("Merge/Rename Clusters"):
            # Copy the original DataFrame.
            df_merged = df.copy()

            # Process each group with the merge_clusters function.
            # Note: merge_clusters should be defined to handle a list of clusters
            # and a new name. If only one cluster is provided, it should simply rename it.
            for group in merge_groups.values():
                clusters_to_process = group["clusters"]
                new_name = group["new_name"]

                df_merged = merge_clusters(df_merged, clusters_to_process, new_name)

            st.session_state.df_merged = df_merged
            st.success("Clusters processed successfully!")
            st.dataframe(df_merged)

            # Optionally, save the merged/renamed DataFrame.
            merged_csv_path = os.path.join(output_dir, 'cluster_data_merged.csv')
            df_merged.to_csv(merged_csv_path, index=False)

        # Show the currently processed data.
        if hasattr(st.session_state, 'df_merged'):
            st.subheader("Current Processed Data")
            st.dataframe(st.session_state.df_merged)
            if st.button("Reset to Original Data"):
                del st.session_state.df_merged
                st.rerun()


with tab3:
    st.subheader("Generate Charts")
    # Add chart options
    use_merged = st.checkbox("Use merged clusters", value=False)
    use_adjusted = st.checkbox("Use adjusted percentages", value=False, 
                              disabled='adjusted_df' not in st.session_state)
    add_trendline = st.checkbox("Add trend lines", value=False)
    
    # Add option for static vs interactive chart
    chart_type = st.radio("Chart Type:", ["Interactive", "Static"])
    
    if st.button("Generate Chart"):
        # Determine which dataframe to use
        plot_df = df  # Default to original
        
        if use_adjusted and 'adjusted_df' in st.session_state:
            plot_df = st.session_state.adjusted_df
        elif use_merged and hasattr(st.session_state, 'df_merged'):
            plot_df = st.session_state.df_merged
        
        # Get y-columns (all percent columns)
        y_columns = [col for col in plot_df.columns if col.endswith('_Percent')]
        
        if chart_type == "Interactive":
            # Create interactive plotly chart
            fig = generate_stacked_bar_chart(
                plot_df,
                'datetime',
                y_columns,
                add_trendline=add_trendline
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create static matplotlib chart
            fig = generate_static_chart(
                plot_df,
                'datetime',
                y_columns
            )
            
            # Display chart
            st.pyplot(fig)

with tab4:
    st.subheader("Export Results")
    # Export options
    use_merged_export = st.checkbox("Use merged data for export", value=False)
    use_adjusted_export = st.checkbox("Use adjusted percentages for export", value=False,
                                    disabled='adjusted_df' not in st.session_state)
    add_trendline_export = st.checkbox("Include trend lines in exported chart", value=False)
    chart_filename = st.text_input("Chart filename", value="cluster_coverage_chart.html")
    
    if st.button("Export Data and Chart"):
        # Determine which dataframe to use for export
        export_df = df  # Default to original
        
        if use_adjusted_export and 'adjusted_df' in st.session_state:
            export_df = st.session_state.adjusted_df
            st.info("Exporting with adjusted cluster percentages")
        elif use_merged_export and hasattr(st.session_state, 'df_merged'):
            export_df = st.session_state.df_merged
            st.info("Exporting with merged clusters")
        
        # Export DataFrame to CSV
        export_csv_path = os.path.join(output_dir, 'final_cluster_data.csv')
        export_df.to_csv(export_csv_path, index=False)
        
        # Generate chart
        y_columns = [col for col in export_df.columns if col.endswith('_Percent')]
        fig = generate_stacked_bar_chart(
            export_df,
            'datetime',
            y_columns,
            add_trendline=add_trendline_export
        )
        
        # Save chart
        chart_path = os.path.join(output_dir, chart_filename)
        fig.write_html(chart_path)
        
        # Create ZIP file with all results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add CSV files
            zf.write(export_csv_path, os.path.basename(export_csv_path))
            
            # Add chart
            zf.write(chart_path, os.path.basename(chart_path))
            
            # Add all segmented images
            for filename in os.listdir(output_dir):
                if filename.startswith("segmented_") and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(output_dir, filename)
                    zf.write(file_path, filename)
        
        # Download button for the ZIP file
        zip_buffer.seek(0)
        st.download_button(
            label="Download All Results",
            data=zip_buffer,
            file_name="segmentation_results.zip",
            mime="application/zip"
        )
        
        st.success("Export complete! Click the button above to download all results.")
    else:
        if upload_method == "Local Directory":
            st.error(f"No valid images found in the specified directory: {input_directory}")
    else:
        st.error("No files were uploaded. Please check your input.")
        st.info("Please upload images or specify a directory path to begin processing.")

# Footer
st.markdown("---")
st.markdown("K-Means Image Segmentation Tool - oceancv.org")