import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
try:
    from skimage.segmentation import slic
    from skimage.util import img_as_float
except ImportError:
    raise ImportError("scikit-image is not installed. Please run 'pip install scikit-image'")
from datetime import datetime
import pandas as pd
import re

# Define a fixed set of 10 contrasting colors (in BGR format for OpenCV)
CONTRASTING_COLORS = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Brown
    (128, 128, 0),   # Teal
    (0, 0, 128)      # Dark Red
]

def extract_datetime(filename):
    """Extract datetime from filename if possible, otherwise try to extract numeric value or use filename as identifier."""
    # Try to extract datetime
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            dt_part = parts[2]
            if dt_part.startswith("20") and "T" in dt_part:  # Simple check for datetime format
                dt_str = dt_part.split('.')[0]  # Remove file extension if present
                return datetime.strptime(dt_str, "%Y%m%dT%H%M%S")
    except Exception as e:
        pass  # Silently fail and try other methods
    
    # Try to extract numeric filename (like 0.jpg, 1.jpg, etc.)
    try:
        # Extract number from beginning of filename
        numeric_match = re.match(r'^(\d+)', os.path.splitext(filename)[0])
        if numeric_match:
            # Return integer for proper numeric sorting
            return int(numeric_match.group(1))
    except Exception as e:
        pass
    
    # Return the filename as a fallback
    return filename

def apply_kmeans(image, n_clusters, kmeans_model=None):
    """Apply KMeans clustering to an image."""
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    if kmeans_model is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(pixel_values)
    else:
        kmeans = kmeans_model

    labels = kmeans.predict(pixel_values)
    segmented_image = kmeans.cluster_centers_[labels]
    segmented_image = segmented_image.reshape(image.shape)
    segmented_image = np.uint8(segmented_image)

    return segmented_image, labels, kmeans

def get_contrasting_colors(n):
    """
    Get a list of pre-defined contrasting colors.
    
    Args:
        n: Number of colors to return
        
    Returns:
        List of BGR color tuples
    """
    # Return only the number of colors needed, up to 10
    return CONTRASTING_COLORS[:min(n, len(CONTRASTING_COLORS))]

def generate_color_swatch(hsv_value, size=(100, 100)):
    """Generate a color swatch image from HSV value."""
    # Parse HSV value from string format like "(H, S, V)"
    if isinstance(hsv_value, str):
        try:
            # Extract values from string like "(120.5, 0.75, 0.85)"
            hsv_parts = hsv_value.replace('(', '').replace(')', '').split(',')
            h, s, v = float(hsv_parts[0]), float(hsv_parts[1]), float(hsv_parts[2])
        except Exception:
            # Default values if parsing fails
            h, s, v = 0, 0, 0
    else:
        h, s, v = hsv_value
    
    # Clamp values to valid ranges for OpenCV
    # H: 0-179 (OpenCV uses 0-179 for hue), S,V: 0-255
    h = min(179, max(0, int(h)))  # Ensure H is in 0-179 range
    s = min(255, max(0, int(s * 255)))  # Scale and clamp S to 0-255
    v = min(255, max(0, int(v * 255)))  # Scale and clamp V to 0-255
    
    # Create a solid color image
    swatch = np.ones((*size, 3), dtype=np.uint8)
    swatch[:] = (h, s, v)
    
    # Convert from HSV to BGR for display
    swatch = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)
    
    return swatch

def create_contrasting_segmented_image(image, labels, n_clusters):
    """Create a segmented image with contrasting colors for each cluster."""
    # Use pre-defined contrasting colors
    colors = get_contrasting_colors(n_clusters)
    
    # Create output image with the same shape as input
    segmented = np.zeros_like(image)
    
    # Reshape labels to match image dimensions
    labels_2d = labels.reshape(image.shape[0], image.shape[1])
    
    # Assign colors to pixels based on cluster labels
    for i in range(n_clusters):
        segmented[labels_2d == i] = colors[i]
    
    return segmented

def preprocess_image(image):
    """Preprocess an image using CLAHE and HSV conversion."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    preprocessed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return preprocessed_image

def segment_image(image, n_segments):
    """Segment an image using SLIC algorithm."""
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=10, start_label=0)
    return segments

def process_image(image_path, output_dir, n_clusters=4, kmeans_model=None):
    """Process a single image and return cluster information."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None
    
    # Extract filename and identifier (datetime or numeric)
    filename = os.path.basename(image_path)
    identifier = extract_datetime(filename)
    
    # Preprocess image
    preprocessed_image = preprocess_image(image)
    
    # Apply KMeans clustering
    segments = segment_image(preprocessed_image, n_segments=500)
    segmented_image, labels, kmeans = apply_kmeans(preprocessed_image, n_clusters=n_clusters, kmeans_model=kmeans_model)
    
    # Create contrasting color version for better visualization
    contrasting_image = create_contrasting_segmented_image(preprocessed_image, labels, n_clusters)
    
    # Calculate cluster information
    label_counts = Counter(labels)
    cluster_info = []
    total_pixels = preprocessed_image.shape[0] * preprocessed_image.shape[1]
    cluster_hsv_values = kmeans.cluster_centers_
    
    # Create data for CSV - use a consistent column name regardless of identifier type
    cluster_data = {'filename': filename, 'datetime': identifier}
    
    # Calculate percentage for each cluster
    for i in range(n_clusters):
        cluster_percentage = (label_counts[i] / total_pixels) * 100
        cluster_hsv = cluster_hsv_values[i]
        cluster_info.append(f"Cluster {i}: {label_counts[i]} pixels ({cluster_percentage:.2f}%) - HSV: ({cluster_hsv[0]:.2f}, {cluster_hsv[1]:.2f}, {cluster_hsv[2]:.2f})")
        cluster_data[f'Cluster_{i}_Percent'] = cluster_percentage
        cluster_data[f'Cluster_{i}_HSV'] = f"({cluster_hsv[0]:.2f}, {cluster_hsv[1]:.2f}, {cluster_hsv[2]:.2f})"
    
    # Save segmented image with cluster information overlay
    output_image = segmented_image.copy()
    for i, info in enumerate(cluster_info):
        text_position = (10, 30 + i * 20)
        cv2.putText(output_image, info, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save the processed image
    output_filename = f"segmented_{filename}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output_image)
    
    # Also save the contrasting version
    contrast_output_filename = f"contrasting_{filename}"
    contrast_output_path = os.path.join(output_dir, contrast_output_filename)
    cv2.imwrite(contrast_output_path, contrasting_image)
    
    return segmented_image, cluster_data, kmeans, contrasting_image

def process_image_folder(input_dir, output_dir, n_clusters=4):
    """Process all images in a folder and generate CSV with cluster data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the input directory.")
        return None, None
    
    # Process the first image to initialize KMeans model
    first_image_path = os.path.join(input_dir, image_files[0])
    _, _, kmeans_model = process_image(first_image_path, output_dir, n_clusters)
    
    # Process all images
    all_cluster_data = []
    processed_images = []
    
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        segmented_image, cluster_data, _, contrasting_image = process_image(
            image_path, output_dir, n_clusters, kmeans_model
        )
        
        if cluster_data:
            all_cluster_data.append(cluster_data)
            processed_images.append((filename, segmented_image, contrasting_image))
    
    # Create a DataFrame from all cluster data
    if all_cluster_data:
        df = pd.DataFrame(all_cluster_data)
        
        # Sort by datetime/identifier for consistent ordering
        if 'datetime' in df.columns:
            # Check if all values are numeric
            if all(isinstance(x, (int, float)) for x in df['datetime'] if not pd.isna(x)):
                df['datetime'] = pd.to_numeric(df['datetime'], errors='coerce')
            # Sort the DataFrame
            df = df.sort_values('datetime')
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'cluster_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Cluster data saved to {csv_path}")
        
        return df, processed_images
    else:
        print("No images were successfully processed.")
        return None, None

def merge_clusters(df, clusters_to_merge, new_cluster_name):
    """
    Merge specified clusters in the DataFrame and rename them.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cluster data
    clusters_to_merge : list
        List of cluster indices to merge
    new_cluster_name : str
        Name for the new merged cluster
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with merged clusters
    """
    # Create a copy of the DataFrame
    df_merged = df.copy()
    
    # Create a new column for the merged cluster
    merge_col_name = f'Cluster_{new_cluster_name}_Percent'
    df_merged[merge_col_name] = 0
    
    # Sum the percentages of the clusters to merge
    for cluster_idx in clusters_to_merge:
        col_name = f'Cluster_{cluster_idx}_Percent'
        if col_name in df_merged.columns:
            df_merged[merge_col_name] += df_merged[col_name]
            # Optionally, drop the original columns
            df_merged = df_merged.drop(columns=[col_name, f'Cluster_{cluster_idx}_HSV'])
    
    return df_merged

def generate_stacked_bar_chart(df, x_column='datetime', y_columns=None, 
                               title='Percent Cover of Clusters Over Time', 
                               add_trendline=False, trendline_clusters=None):
    """
    Generate a stacked bar chart from DataFrame with optional trend lines.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cluster data
    x_column : str
        Column to use for x-axis
    y_columns : list or None
        Columns to include in the stacked bar chart. If None, all columns ending with '_Percent' are used.
    title : str
        Title for the chart
    add_trendline : bool
        Whether to add trend lines to the chart
    trendline_clusters : list or None
        List of clusters to add trend lines for. If None and add_trendline is True, 
        trend lines are added for all clusters.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The generated figure
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # If y_columns not specified, use all percent columns
    if y_columns is None:
        y_columns = [col for col in df.columns if col.endswith('_Percent')]
    
    # Convert x-axis values for proper sorting if they're numeric or dates
    x_values = df[x_column].copy()
    numeric_x = False
    
    # Check if x values are datetime objects but stored as strings
    if isinstance(x_values.iloc[0], str) and any([":" in str(x) for x in x_values]):
        try:
            x_values = pd.to_datetime(x_values)
        except:
            pass
    
    # If x values are numeric or datetime, sort the dataframe
    if pd.api.types.is_numeric_dtype(x_values) or isinstance(x_values.iloc[0], datetime):
        df = df.sort_values(x_column)
        numeric_x = True
    
    fig = go.Figure()
    
    # Add each cluster as a bar
    for column in y_columns:
        # Extract cluster name for the legend
        cluster_name = column.replace('_Percent', '')
        fig.add_trace(go.Bar(
            x=df[x_column],
            y=df[column],
            name=cluster_name
        ))
        
        # Add trend line if requested
        if add_trendline and (trendline_clusters is None or cluster_name in trendline_clusters):
            # Only add trend lines for numeric x-values or datetime
            if numeric_x:
                # Convert to numeric for trend calculation
                if isinstance(df[x_column].iloc[0], datetime):
                    # Convert datetime to unix timestamp for calculation
                    x_numeric = np.array([(d - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') 
                                         for d in df[x_column]])
                else:
                    x_numeric = df[x_column].values
                
                # Calculate trendline
                y = df[column].values
                mask = ~np.isnan(x_numeric) & ~np.isnan(y)
                if np.sum(mask) > 1:  # Need at least 2 points for a line
                    slope, intercept = np.polyfit(x_numeric[mask], y[mask], 1)
                    y_trend = slope * x_numeric + intercept
                    
                    # Add the trendline
                    fig.add_trace(go.Scatter(
                        x=df[x_column],
                        y=y_trend,
                        mode='lines',
                        name=f'{cluster_name} trend',
                        line=dict(dash='dash'),
                        opacity=0.7
                    ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title='Time',
        yaxis_title='Percent Cover',
        template='plotly_white',
        hovermode='x unified',
        legend_title='Clusters'
    )
    
    return fig
