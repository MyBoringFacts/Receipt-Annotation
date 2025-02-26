# Receipt Processing and Annotation Pipeline

This repository provides an end-to-end pipeline for processing receipt images using Azure Document Intelligence. It extracts key fields from receipts, converts them into YOLO-format annotations for object detection, and then transforms these annotations into LabelMe format for manual review or further processing.

## Overview

The pipeline consists of the following steps:

1. **Image Renaming**  
   - Standardizes image filenames in the `images/` folder (e.g., `0.jpg`, `1.jpg`, etc.) to ensure consistency.

2. **Receipt Extraction with Azure Document Intelligence**  
   - Processes each image using Azure's prebuilt receipt model.
   - Saves the extracted JSON results in the `images_json/` folder.

3. **YOLO Annotation Generation**  
   - Converts JSON outputs into YOLO annotation format by extracting key fields such as:
     - Seller Name (Class 0)
     - Seller VAT Number (Class 1)
     - Document Date (Class 2)
     - Product Description (Class 3)
     - Quantity (Class 4)
     - Price (Class 5)
     - Total Due Amount (Class 6)
   - Saves these annotations into the `yolo_annotations/` folder.

4. **Dataset Organization for YOLO Training**  
   - Splits the dataset into training (80%) and validation (20%) sets.
   - Organizes images and labels in a structure compatible with YOLO training.
   - Generates a `data.yaml` file with paths, number of classes, and class names.

5. **Conversion to LabelMe Format**  
   - Converts YOLO annotations back to LabelMe JSON format.
   - Embeds the image (encoded in base64) along with its annotation data.
   - Saves the LabelMe JSON files into the `labelme_data/` folder.



## Requirements

- Python 3.7+
- [Azure AI Document Intelligence SDK](https://pypi.org/project/azure-ai-documentintelligence/)
- OpenCV (`opencv-python`)
- scikit-learn

Install the dependencies using:

```bash
pip install azure-ai-documentintelligence opencv-python scikit-learn
```

## Configuration

### Azure Credentials

In the script, update the following variables with your Azure Document Intelligence endpoint and API key:

```python
endpoint = "YOUR_AZURE_ENDPOINT"
key = "YOUR_API_KEY"
```

### Folder Paths

Adjust the paths if needed:

- Input images: `images/`
- JSON outputs: `images_json/`
- YOLO annotations: `yolo_annotations/`
- YOLO dataset: `yolo_dataset/`
- LabelMe annotations: `labelme_data/`

## Usage

Run the main script to execute the entire pipeline:

```bash
python main.py
```

This will:
- Rename images in the `images/` folder.
- Process images with Azure Document Intelligence and save JSON results.
- Generate YOLO annotations from the JSON files.
- Split the dataset into training and validation sets.
- Convert YOLO annotations into LabelMe format.
- labelme part is optional. you can delete that part of code if you dont need it.

## Customization and Future Enhancements

- **Parallel Processing:**  
  Consider using asynchronous requests or multiprocessing to speed up receipt processing via Azure Document Intelligence.

- **Dynamic File Handling:**  
  Enhance file handling to automatically detect and process multiple image formats.

- **Enhanced Logging & Monitoring:**  
  Integrate a logging system or dashboard to monitor pipeline progress and diagnose errors in real time.

- **Workflow Orchestration:**  
  Use workflow managers like Apache Airflow or Prefect to orchestrate and schedule pipeline tasks.

- **User Interface:**  
  Develop an interactive UI for manual review and adjustment of annotations in LabelMe.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions and improvements.
