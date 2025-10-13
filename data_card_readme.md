# GUI-360°: A Comprehensive Dataset And Benchmark For Computer-Using Agents

## Introduction

We introduce GUI-360°, a large-scale, comprehensive dataset and benchmark
suite designed to advance computer-using agents (CUAs). CUAs present unique
challenges and is constrained by three persistent gaps: a scarcity of real-world CUA tasks, the lack of automated collection-and-annotation pipelines for multi-modal trajectories, and the absence of a unified benchmark that jointly evaluates GUI grounding, screen parsing, and action prediction. GUI-360° addresses these gaps with a largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering. The released corpus contains over 1.2M executed action steps across thousands of trajectories in popular Windows office applications, and includes full-resolution screenshots, accessibility metadata when available, instantiated goals, intermediate reasoning traces, and both successful and
failed action trajectories. The dataset supports three canonical tasks, GUI grounding, screen parsing, and action prediction, and a hybrid GUI+API action space that reflects modern agent designs. Benchmarking state-of-the-art vision–language models on GUI-360◦ reveals substantial out-of-the-box shortcomings in grounding and action prediction; supervised fine-tuning yield significant gains.

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

For example:
```json
{
        "id": "excel_4s_1_2",
        "images": [
            "images\\excel_4s_1/action_step4.png"
        ],
        "conversation": [
            {
                "from": "human",
                "value": "<image>\nYou are a helpful assistant. Given a screenshot of the current screen and user instruction, you need to output the position of the element you will operate.\n\nThe instruction is:\nTo use the ATAN function as requested, I will enter the formula =ATAN(1) into cell H4, which is currently empty and selected. This will calculate the arctangent of 1 as a placeholder, as the user has not specified a different value.\n\nOutput the coordinate of the element you will operate within <coordinate></coordinate> tag:\n<coordinate> [x, y] </coordinate>\n"
            },
            {
                "from": "gpt",
                "value": "<coordinate> [630, 241] </coordinate>"
            }
        ]
    }
```

### Screen Parsing

- **Goal**: Identify and extract information about all interactive UI elements from a screenshot.
- **Input**:
    - `step.screenshot_clean`: The screenshot of the application.
- **Output**:
    - `step.control_infos`: A collection of information for all UI controls visible on the screen.

For example:

```json
{
    "id": "word_bing_search_word_4s_1486_1",
    "images": [
      "images/word_4s_1486/action_step1.png"
    ],
    "conversation": [
      {
        "from": "human",
        "value": "<image>\n\nYou are an expert in screen parsing and GUI element extraction.\n\nYou will be provided with a screenshot of a desktop application interface. Your task is to find all interactive controls on the screen (anywhere that can be clicked, typed into, etc.), with a maximum number not to exceed 500.\n(If it's Excel, then interactive cells also count as controls.)\n\nFor each control element you identify, you need to provide:\n\n1. **control_text**: The visible text displayed on or within the control. This includes button labels, input field placeholders, menu item names, checkbox labels, tab titles, etc. Use empty string \"\" if the control has no visible text content.\n2. **control_rect**: The bounding rectangle coordinates listed as [left, top, right, bottom] in absolute pixel coordinates, relative to the top-left corner of the screenshot.\n\nOutput your response in JSON format as a list of control information objects:\n\n```json\n[\n  {\n    \"control_text\": \"Macro Recording Not Recording\",\n    \"control_rect\": [54, 699, 86, 720]\n  },\n  {\n    \"control_text\": \"Accessibility Checker Accessibility: Good to go\",\n    \"control_rect\": [86, 699, 247, 720]\n  },\n  {\n    \"control_text\": \"Page Layout\",\n    \"control_rect\": [764, 699, 804, 720]\n  },\n  {\n    \"control_text\": \"Zoom In\",\n    \"control_rect\": [916, 699, 963, 720]\n  },\n  ...(more controls exhaustively listed)...\n]\n```\n\nImportant guidelines:\n- ONLY include interactive controls that are currently enabled and active\n- EXCLUDE disabled, grayed-out, or non-interactive elements\n- EXCLUDE static text, labels, and decorative elements that cannot be clicked\n- Focus on controls users can actually interact with (click, type, select, etc.)\n- Include empty string for control_text if there's no visible text\n- Provide accurate bounding rectangles in pixel coordinates\n- Be thorough but selective - only active interactive elements\n- Exhaustively identify all interactive controls on the screen, and do not miss any\n- Ensure the JSON output is syntactically correct.\n- Please observe and inspect the screenshot carefully, and think very carefully before you answer internally. But you do not need to write out your thoughts in the final answer, only output the JSON array.\n\nYour response should contain ONLY the JSON array, no additional text or explanations.\n"
      },
      {
        "from": "gpt",
        "value": "[{\"control_type\": \"Button\", \"control_rect\": [50, 17, 161, 46], \"control_text\": \"AutoSave\", \"source\": \"uia\", \"label\": 1}, {\"control_type\": \"Button\", \"control_rect\": [162, 17, 190, 46], \"control_text\": \"Save\", \"source\": \"uia\", \"label\": 2}]"
      }
    ]
  }
```

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

For example:

```json
{
    "id": "excel_bing_search_excel_4s_1_3",
    "images": [
      "images/excel_4s_1/action_step5.png"
    ],
    "conversation": [
      {
        "from": "human",
        "value": "<image>\nYou are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.\n\nThe instruction is:\nEnable editing and use the ATAN function in Excel to calculate the arctangent of a number.\n\nThe history of actions are:\nStep 1: To use the ATAN function as requested, I will enter the formula =ATAN(1) into cell H4, which is currently empty and selected. This will calculate the arctangent of 1 as a placeholder, as the user has not specified a different value.\n\nThe actions supported are:\n<action>\n- click\n  - Args:\n    - coordinate: [x, y], the absolute position on the screen you want to click at.\n    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')\n    - double: bool, Whether to perform a double click or not (Default: False)'\n    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)\n  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')\n- type\n  - Args:\n    - coordinate: [x, y], the absolute position on the screen you want to type at.\n    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, \"{VK_CONTROL}c\" represents the Ctrl+C shortcut key.\n    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)\n    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) \n  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys=\"{TAB 2}\")\n- drag\n  - Args:\n    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.\n    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.\n    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')\n    - duration: float, The duration of the drag action in seconds. (Default: 1.0)\n    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)\n  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')\n- wheel_mouse_input\n  - Args: \n    - coordinate: [x, y], the absolute position on the screen to scroll.\n    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.\n  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)\n- table2markdown\n  - Args:\n    - sheet_name: str|int, The name or index of the sheet to get the table content. The index starts from 1.\n  - Example: table2markdown(sheet_name=1)\n- insert_excel_table\n  - Args:\n    - table: list[list], The table content to insert. The table is a list of list of strings or numbers.\n    - sheet_name: str, The name of the sheet to insert the table.\n    - start_row: int, The start row to insert the table, starting from 1.\n    - start_col: int, The start column to insert the table, starting from 1.\n  - Example: insert_excel_table(table=[[\"Name\", \"Age\", \"Gender\"], [\"Alice\", 30, \"Female\"], [\"Bob\", 25, \"Male\"], [\"Charlie\", 35, \"Male\"]], sheet_name=\"Sheet1\", start_row=1, start_col=1)\n- select_table_range\n  - Args:\n    - sheet_name: str, The name of the sheet.\n    - start_row: int, The start row, starting from 1.\n    - start_col: int, The start column, starting from 1.\n    - end_row: int, The end row. If ==-1, select to the end of the document with content.\n    - end_col: int, The end column. If ==-1, select to the end of the document with content.\n  - Example: select_table_range(sheet_name=\"Sheet1\", start_row=1, start_col=1, end_row=3, end_col=3)\n- set_cell_value\n  - Args:\n    - sheet_name: str, The name of the sheet.\n    - row: int, The row number (1-based).\n    - col: int, The column number (1-based).\n    - value: str|int|float|None, The value to set in the cell. If None, just select the cell.\n    - is_formula: bool, If True, treat the value as a formula, otherwise treat it as a normal value. (Default: False)\n  - Example: set_cell_value(sheet_name=\"Sheet1\", row=1, col=1, value=\"Hello\", is_formula=False), set_cell_value(sheet_name=\"Sheet1\", row=2, col=2, value=\"=SUM(A1:A10)\", is_formula=True)\n- auto_fill\n  - Args:\n    - sheet_name: str, The name of the sheet.\n    - start_row: int, The starting row number (1-based).\n    - start_col: int, The starting column number (1-based).\n    - end_row: int, The ending row number (1-based).\n    - end_col: int, The ending column number (1-based).\n  - Example: auto_fill(sheet_name=\"Sheet1\", start_row=1, start_col=1, end_row=10, end_col=3)\n- reorder_columns\n  - Args:\n    - sheet_name: str, The name of the sheet.\n    - desired_order: list[str], The list of column names in the new order.\n  - Example: reorder_columns(sheet_name=\"Sheet1\", desired_order=[\"Income\", \"Date\", \"Expense\"]) \n</action>\nImportant: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)\n\nFirst, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.\n\nThen output your action within <tool_call></tool_call> tag like:\n<tool_call>\n{\n  \"function\": \"<function name>\",\n  \"args\": {},\n  \"status\": \"CONTINUE\"\n}\n</tool_call>\n\nIf you think the task is finished, you can output status as \"FINISH\" and take no action. Like:\n<tool_call>\n{\n  \"function\": \"\",\n  \"args\": {},\n  \"status\": \"FINISH\"\n}\n</tool_call>\n\nOnly **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.\n"
      },
      {
        "from": "gpt",
        "value": "<tool_call>\n{\n    \"function\": \"type\",\n    \"args\": {\n        \"keys\": \"{ENTER}\",\n        \"control_focus\": true,\n        \"coordinate\": [\n            628,\n            239\n        ]\n    },\n    \"status\": \"CONTINUE\"\n}\n</tool_call>"
      }
    ]
  }
```

The repository is organized as follows:

```
data/
└── train(test)/
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