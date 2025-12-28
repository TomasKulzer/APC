Repository for the curricular unit APC (Machine Learning)

## Setup Instructions

### 1. Create a Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Code

Run each script in order (00, 01, 02, etc.).

**Note:** To use only color histogram features, do not run the HOG or SIFT scripts or delete their features from the features/ directory.

### Deactivating the Virtual Environment
When finished, you can deactivate the virtual environment:
```bash
deactivate
```