# K-Means Image Segmentation

This project provides a tool for segmenting images using the K-Means clustering algorithm. It allows users to process images, analyze cluster distributions, merge similar clusters, and generate time-series charts.

## Features

*   **Image Segmentation:** Utilizes K-Means clustering to segment images based on color similarity.
*   **Cluster Analysis:** Provides information about the pixel distribution and HSV values for each cluster.
*   **Cluster Merging:** Allows users to merge similar clusters to simplify the segmentation results.
## Running the Streamlit App Locally

To run the Streamlit application locally, follow these steps:

1.  **Install Dependencies:** Make sure you have all the required Python packages installed. You can install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:** Navigate to the directory containing `app.py` and run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3.  **Access the App:** Open your web browser and go to the address displayed in the console (usually `http://localhost:8501`).

## File Descriptions

*   `image_segmentation.py`: Contains the core image segmentation functions.
*   `app.py`: Implements the Streamlit application.
*   `snow_segmentation_notebook.ipynb`: A Jupyter Notebook demonstrating the usage of the image segmentation functions.
*   `requirements.txt`: Lists the required Python packages.
*   `LICENSE`: Contains the license information.
*   `README.md`: This file, providing an overview of the project.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.