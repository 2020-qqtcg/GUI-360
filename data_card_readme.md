# GUI-360°: A Comprehensive Dataset And Benchmark For Computer-Using Agents

## Introduction

We introduce GUI-360°, a large-scale, comprehensive dataset and benchmark
suite designed to advance computer-using agents (CUAs). CUAs present unique
challenges and is constrained by three persistent gaps: a scarcity of real-world CUA tasks, the lack of automated collection-and-annotation pipelines for multi-modal trajectories, and the absence of a unified benchmark that jointly evaluates GUI grounding, screen parsing, and action prediction. GUI-360° addresses these gaps with a largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering. The released corpus contains over 1.2M executed action steps across thousands of trajectories in popular Windows office applications, and includes full-resolution screenshots, accessibility metadata when available, instantiated goals, intermediate reasoning traces, and both successful and
failed action trajectories. The dataset supports three canonical tasks, GUI grounding, screen parsing, and action prediction, and a hybrid GUI+API action space that reflects modern agent designs. Benchmarking state-of-the-art vision–language models on GUI-360◦ reveals substantial out-of-the-box shortcomings in grounding and action prediction; supervised fine-tuning yield significant gains.

## Github

We provide comprehensive tools for processing the raw dataset and evaluating model performance on GUI-360°: 
- **https://github.com/2020-qqtcg/GUI-360**

## Data Structure

Each data sample includes the following fields:

```json
{
  "execution_id": "string",           // Unique execution identifier: {app}_{tag}_{id}
  "app_domain": "string",             // Application domain: excel/word/ppt
  "request": "string",                // Natural language description of the user request
  "template": "string",               // Template file name used
  "step_id": "number",                // Current step ID
  "total_steps": "number",            // Total number of steps
  "evaluation": {                     // Task evaluation results
    "reason": "string",               // Reason for the evaluation
    "evidence": "string",             // Evidence for the evaluation
    "sub_scores": {},                 // Sub-task scores
    "complete": "yes/no"              // Whether the task was completed
  },
  "step": {                          // Detailed step information
    "screenshot_clean": "string",     // Path to the clean screenshot
    "screenshot_desktop": "string",   // Path to the desktop screenshot
    "screenshot_annotated": "string", // Path to the annotated screenshot
    "screenshot_selected_controls": "string", // Path to the screenshot of selected controls
    "ui_tree": {},                    // UI tree structure
    "control_infos": {                // Control information
      "application_windows_info": {}, // Application window information
      "uia_controls_info": []
    },
    "subtask": "string",              // Description of the sub-task
    "observation": "string",          // Observation result
    "thought": "string",              // Thought process
    "action": {                       // Action performed
      "action_type": "GUI/API",           // Type of action
      "control_text": "string",       // Control text
      "control_label": "string",      // Control label
      "function": "string",           // Function executed (e.g., click)
      "args": {},                     // Function arguments
      "rectangle": {},                // Control's bounding rectangle
      "coordinate_x": "number",       // X-coordinate
      "coordinate_y": "number",       // Y-coordinate
      "desktop_rectangle": {},        // Bounding rectangle on the desktop
      "desktop_coordinate_x": "number", // Desktop X-coordinate
      "desktop_coordinate_y": "number"  // Desktop Y-coordinate
    },
    "status": "CONTINUE/FINISH/OVERALL_FINISH",    // Execution status
    "tags": [],         // Support task type [grounding, action_prediction, screen_parsing]
  }
}
```

On this basis, we processed GUI-360° into three types of tasks:
- Grounding
- Screen Parsing
- Action Prediction

### Grounding
- **Goal**: Locate the position of a UI element based on an image and a natural language instruction.
- **Input**: 
    - `step.screenshot_clean`: The screenshot of the application.
    - `step.thought`: The natural language instruction describing the element to find.
- **Output**:
    - `step.action.coordinate_x`, `step.action.coordinate_y`: The coordinates of the target UI element.
    - **Evaluation**: The evaluation is based on whether the predicted coordinates fall within the ground-truth rectangle.


### Screen Parsing

- **Goal**: Identify and extract information about all interactive UI elements from a screenshot.
- **Input**:
    - `step.screenshot_clean`: The screenshot of the application.
- **Output**:
    - `step.control_infos`: A collection of information for all UI controls visible on the screen.


### Action Prediction

- **Goal**: Predict the next action to take based on the current state and overall goal.
- **Input (with screenshot)**:
    - `step.screenshot_clean`: The screenshot of the application.
    - `request`: The high-level user request for the entire task.
    - action history.
- **Input (with screenshot + a11y)**:
    - `step.screenshot_annotated`: The annotated screenshot of the application.
    - `step.ui_tree`: The accessibility tree of the current view.
    - `request`: The high-level user request for the entire task.
    - action history.
- **Output**:
    - `step.action`: The predicted action to be performed next.


## Data Organization

GUI-360° data organization structure:

- **Base data**: Stored in `train`, `test` and `fail` directories
- **Processed data**: Processed data stored in `processed` directory  
- **Template files**: All templates used are provided in `template` directory

### train/test/fail Directory Structure

```
data/
└── train(test/fail)/
    ├── data/
    │   ├── excel/
    │   │   ├── qabench/success/     # Excel QABench tasks
    │   │   ├── bing_search/success/ # Excel Bing Search tasks  
    │   │   └── m365/success/        # Excel M365 tasks
    │   ├── word/
    │   │   ├── qabench/success/     # Word QABench tasks
    │   │   ├── bing_search/success/ # Word Bing Search tasks
    │   │   ├── m365/success/        # Word M365 tasks
    │   │   └── wikihow/success/     # Word WikiHow tasks
    │   └── ppt/
    │       ├── qabench/success/     # PowerPoint QABench tasks
    │       ├── bing_search/success/ # PowerPoint Bing Search tasks
    │       └── m365/success/        # PowerPoint M365 tasks
    └── image/
        ├── excel/
        │   ├── qabench/success/     # Excel QABench tasks
        │   ├── bing_search/success/ # Excel Bing Search tasks  
        │   └── m365/success/        # Excel M365 tasks
        ├── word/
        │   ├── qabench/success/     # Word QABench tasks
        │   ├── bing_search/success/ # Word Bing Search tasks
        │   ├── m365/success/        # Word M365 tasks
        │   └── wikihow/success/     # Word WikiHow tasks
        └── ppt/
            ├── qabench/success/     # PowerPoint QABench tasks
            ├── bing_search/success/ # PowerPoint Bing Search tasks
            └── m365/success/        # PowerPoint M365 tasks
```

### Processed Data Directory Structure

```
processed/
├── action_prediction_train_resize/     # Action prediction training data
├── action_prediction_train_resize_a11y/ # Action prediction training data (with accessibility info)
├── grounding_resize/                   # Grounding task data
└── screen_parsing_train_resize/        # Screen parsing training data
```
