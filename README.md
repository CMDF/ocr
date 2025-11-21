# OCR Document Analysis Service

## Project Description

This project implements a sophisticated document analysis service designed to extract structured information from PDF documents, specifically focusing on identifying and linking textual references to figures and tables within the document. Leveraging advanced OCR, layout analysis, and graph-based matching, the service provides a comprehensive understanding of document content and its internal structure.

## Table of Contents

- [Features](#features)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)

## Features

 - **Automated PDF Processing**: Accepts PDF documents via a pre-signed S3 URL for seamless integration.
 - **High-Resolution Image Conversion**: Converts PDF pages into high-resolution PNG images for accurate downstream processing.
 - **Intelligent Layout Detection**: Utilizes PaddleOCR's layout analysis to identify and categorize document elements (text, figures, tables, titles).
 - **Advanced Text Extraction & Correction**: Performs OCR on text regions, followed by robust spelling and segmentatio correction using SymSpellPy.
 - **ML-powered Reference Identification**: Employs machine learning models to detect textual references to figures and tables within the document body.
 - **Graph-based Reference Matching**: Builds a comprehensive document graph to establish hierarchical, sequential, and spatial relationships between elements, enabling precise linking of text references to their corresponding figures/tables.
 - **Structured JSON Output**: Delivers all extracted and linked information in a clean, structured JSON format.

## Pipeline Overview

The core of this service is a multi-stage pipeline designed for in-depth document understanding. The process orchestrates various components including image processing, optical character recognition, natural language processing and graph theory to achieve its goals.

![Image](https://github.com/user-attachments/assets/5b6e8f24-78fe-43fc-966e-c16a4af6d8e2)
_A visual representation of the document analysis pipeline, from PDF input to structured JSON output._

### Detailed Pipeline Steps:

1. **File Download**: Securely downloads the PDF document from a provided S3 pre-signed URL.
2. **PDF to Image Conversion**: Each page of the PDF is converted into a high-resolution PNG image, processed in parallel.
3. **Layout Analysis**: An AI model analyzes the document's layout, identifying and categorizing elements like text blocks, figures, and tables. Pre-processing steps remove nested boxes and intelligently group images with their captions.
4. **OCR & Text Correction**: Text regions are subjected to Optical Character Recognition (OCR), and the extracted text is then spell-checked and corrected for segmentation errors.
5. **Reference Identification**: Machine learning models scan the corrected text to identify phrases that refer to figures or tables (e.g., "as shown in Figure 3").
6. **Document Graph Construction**: All identified elements (text blocks, figures, tables, titles) and their relationships are used to build a comprehensive document graph.
7. **Reference Matching**: The graph is traversed to accurately link the identified textual references to their corresponding figures or tables, resolving ambiguities based on document structure and proximity.
8. **Structured Output**: The final output is a JSON object containing the extracted text, figure information, and the matched text-to-figure references.

## Installation

This section will detail how to set up the project. Since this is an API service, it would typically involve:

1.  **Clone the repository:**
```

git clone https://github.com/CMDF/ocr.git
cd ocr

```
2. **Create a virtual environment (recommended):**
```

python -m venv venv_name
source venv_name/bin/activate

```
3. **Install dependencies:**
(Install PaddlePaddle refer to [Installation Guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html) first.)
```

pip install -r requirements.txt
python -m spacy download en_core_web_sm

```
