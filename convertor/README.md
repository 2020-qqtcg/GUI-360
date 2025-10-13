# 🔄 Data Converter for GUI-360°

This directory contains tools to convert GUI-360° raw dataset into task-specific training formats suitable for vision-language model fine-tuning.

## 📁 Files

- **`convert_to_train.py`** - Main conversion script with CLI interface
- **`prompt_action_prediction.py`** - Action prediction task prompts and templates
- **`prompt_screen_parsing.py`** - Screen parsing task prompts and templates
- **`prompt_grounding.py`** - GUI grounding task prompts and templates

## 🎯 Supported Conversion Tasks

| Task | Description | Output Format |
|------|-------------|---------------|
| **`action_prediction`** | Convert to action prediction training data | Multi-turn conversations with screenshots |
| **`action_prediction_a11y`** | Action prediction with accessibility tree information | Multi-turn conversations with annotated screenshots |
| **`screen_parsing`** | Screen parsing for UI element extraction | Image-to-JSON control extraction |
| **`grounding`** | GUI grounding for element localization | Image-to-coordinates mapping |

## 🚀 Quick Start

### Basic Usage

```bash
# Convert to action prediction format
python convert_to_train.py \
    --root_dir /path/to/GUI360_dataset \
    --output_dir /path/to/output \
    --type action_prediction

# Convert with image resizing (recommended for training)
python convert_to_train.py \
    --root_dir /path/to/GUI360_dataset \
    --output_dir /path/to/output \
    --type action_prediction \
    --resize 
```

### Advanced Examples

**🎯 Action Prediction with A11y**
```bash
python convert_to_train.py \
    --root_dir /path/to/GUI360_dataset \
    --output_dir /path/to/output \
    --type action_prediction_a11y \
    --resize 
```

**🔍 Screen Parsing**
```bash
python convert_to_train.py \
    --root_dir /path/to/GUI360_dataset \
    --output_dir /path/to/output \
    --type screen_parsing \
    --resize 
```

**📍 GUI Grounding**
```bash
python convert_to_train.py \
    --root_dir /path/to/GUI360_dataset \
    --output_dir /path/to/output \
    --type grounding \
    --resize \
    --max_pixels 500000
```

**⚠️ Process Failed Samples Only**
```bash
python convert_to_train.py \
    --root_dir /path/to/GUI360_dataset \
    --output_dir /path/to/output \
    --type action_prediction \
    --success_or_fail fail \
    --verbose
```

## ⚙️ Command Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--root_dir` | `str` | Root directory of the dataset (contains `data/` and `image/` folders) |
| `--output_dir` | `str` | Output directory for processed data and images |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--type` | `str` | `action_prediction` | Conversion type: `action_prediction`, `action_prediction_a11y`, `screen_parsing`, `grounding` |
| `--success_or_fail` | `str` | `success` | Process `success` or `fail` samples |
| `--resize` | `flag` | `False` | Enable image resizing and coordinate scaling |
| `--max_pixels` | `int` | `999999` | Maximum pixels for resized images |
| `--factor` | `int` | `28` | Factor that image dimensions must be divisible by |
| `--output_filename` | `str` | `None` | Custom output JSON filename (auto-generated if not specified) |
| `--verbose` | `flag` | `False` | Enable verbose output |

## 📊 Output Format

### Action Prediction
```json
{
  "id": "excel_4s_1_2",
  "images": ["images/excel_4s_1/action_step2.png"],
  "conversation": [
    {
      "from": "human",
      "value": "<image>\nInstruction: Enable editing and use ATAN function...\nHistory: Step 1: ..."
    },
    {
      "from": "gpt", 
      "value": "<tool_call>\n{\"function\": \"click\", \"args\": {\"coordinate\": [630, 241]}, \"status\": \"CONTINUE\"}\n</tool_call>"
    }
  ]
}
```

### Screen Parsing
```json
{
  "id": "word_4s_1486_1",
  "images": ["images/word_4s_1486/action_step1.png"],
  "conversation": [
    {
      "from": "human",
      "value": "<image>\nFind all interactive controls on the screen..."
    },
    {
      "from": "gpt",
      "value": "[{\"control_text\": \"Save\", \"control_rect\": [162, 17, 190, 46]}]"
    }
  ]
}
```

### GUI Grounding
```json
{
  "id": "excel_4s_1_2", 
  "images": ["images/excel_4s_1/action_step4.png"],
  "conversation": [
    {
      "from": "human",
      "value": "<image>\nLocate the element: Click the 'Bold' button in toolbar"
    },
    {
      "from": "gpt",
      "value": "<coordinate> [450, 120] </coordinate>"
    }
  ]
}
```

## 🖼️ Image Processing

### Smart Resize Feature
- **Purpose**: Optimize images for VLM training while preserving aspect ratios
- **Parameters**:
  - `max_pixels`: Limit total pixel count (e.g., 500,000 for efficiency)
  - `factor`: Ensure dimensions divisible by factor (e.g., 28 for model compatibility)
- **Coordinate Scaling**: Automatically adjusts coordinates when images are resized

### Dependencies
- **Required**: `PIL` (Pillow) for image processing
- **Optional**: `qwen_vl_utils` for optimized smart resize (falls back to built-in implementation)
- **Optional**: `tqdm` for progress bars

## 📋 Dataset Structure

### Input Directory Structure
```
dataset_root/
├── data/
│   ├── excel/
│   │   ├── category1/
│   │   │   └── success/
│   │   │       ├── trajectory1.jsonl
│   │   │       └── trajectory2.jsonl
│   ├── word/
│   └── ppt/
└── image/
    ├── excel/
    │   ├── category1/
    │   │   └── success/
    │   │       ├── image1.png
    │   │       └── image2.png
    ├── word/
    └── ppt/
```

### Output Directory Structure
```
output_dir/
├── images/
│   ├── excel/
│   ├── word/
│   └── ppt/
└── action_prediction_training_data.json  # or other task-specific filename
```
