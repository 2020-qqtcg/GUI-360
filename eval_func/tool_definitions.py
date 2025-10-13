"""
Tool parameter definitions with default values.
Based on the tool descriptions in prompt_action_prediction.py
"""

# Tool parameter definitions with default values
TOOL_DEFAULTS = {
    "click": {
        "coordinate": [0, 0],  # Required parameter, no real default
        "button": "left",
        "double": False,
        "pressed": None
    },
    "type": {
        "coordinate": [0, 0],  # Required parameter, no real default
        "keys": "",  # Required parameter, no real default
        "clear_current_text": False,
        "control_focus": True
    },
    "drag": {
        "start_coordinate": [0, 0],  # Required parameter, no real default
        "end_coordinate": [0, 0],  # Required parameter, no real default
        "button": "left",
        "duration": 1.0,
        "key_hold": None
    },
    "wheel_mouse_input": {
        "coordinate": [0, 0],  # Required parameter, no real default
        "wheel_dist": 0  # Required parameter, no real default
    },
    # Additional tools for Word
    "insert_table": {
        "rows": 1,
        "columns": 1
    },
    "select_text": {
        "text": ""  # Required parameter, no real default
    },
    "select_table": {
        "number": 1
    },
    "select_paragraph": {
        "start_index": 1,
        "end_index": -1,
        "non_empty": True
    },
    "save_as": {
        "file_dir": "",
        "file_name": "",
        "file_ext": ".pdf"
    },
    "set_font": {
        "font_name": None,
        "font_size": None
    },
    # Additional tools for Excel
    "table2markdown": {
        "sheet_name": 1
    },
    "insert_excel_table": {
        "table": [],  # Required parameter, no real default
        "sheet_name": "Sheet1",
        "start_row": 1,
        "start_col": 1
    },
    "select_table_range": {
        "sheet_name": "Sheet1",
        "start_row": 1,
        "start_col": 1,
        "end_row": -1,
        "end_col": -1
    },
    "set_cell_value": {
        "sheet_name": "Sheet1",
        "row": 1,
        "col": 1,
        "value": None,
        "is_formula": False
    },
    "auto_fill": {
        "sheet_name": "Sheet1",
        "start_row": 1,
        "start_col": 1,
        "end_row": 1,
        "end_col": 1
    },
    "reorder_columns": {
        "sheet_name": "Sheet1",
        "desired_order": []  # Required parameter, no real default
    },
    # Additional tools for PowerPoint
    "set_background_color": {
        "color": "FFFFFF",
        "slide_index": None
    }
}

# A11Y tool parameter definitions with default values (prioritizes control_label over coordinate)
TOOL_DEFAULTS_A11Y = {
    "click": {
        "control_label": None,  # Primary parameter for A11Y, use this first
        "coordinate": None,  # Fallback parameter when control_label is not available
        "button": "left",
        "double": False,
        "pressed": None
    },
    "type": {
        "control_label": None,  # Primary parameter for A11Y, use this first
        "coordinate": None,  # Fallback parameter when control_label is not available
        "keys": "",  # Required parameter, no real default
        "clear_current_text": False,
        "control_focus": True
    },
    "drag": {
        "start_coordinate": [0, 0],  # Required parameter, no real default (drag always uses coordinates)
        "end_coordinate": [0, 0],  # Required parameter, no real default (drag always uses coordinates)
        "button": "left",
        "duration": 1.0,
        "key_hold": None
    },
    "wheel_mouse_input": {
        "control_label": None,  # Primary parameter for A11Y, use this first
        "coordinate": None,  # Fallback parameter when control_label is not available
        "wheel_dist": 0  # Required parameter, no real default
    },
    # Additional tools for Word (A11Y versions have same defaults as regular versions)
    "insert_table": {
        "rows": 1,
        "columns": 1
    },
    "select_text": {
        "text": ""  # Required parameter, no real default
    },
    "select_table": {
        "number": 1
    },
    "select_paragraph": {
        "start_index": 1,
        "end_index": -1,
        "non_empty": True
    },
    "save_as": {
        "file_dir": "",
        "file_name": "",
        "file_ext": ".pdf"
    },
    "set_font": {
        "font_name": None,
        "font_size": None
    },
    # Additional tools for Excel (A11Y versions have same defaults as regular versions)
    "table2markdown": {
        "sheet_name": 1
    },
    "insert_excel_table": {
        "table": [],  # Required parameter, no real default
        "sheet_name": "Sheet1",
        "start_row": 1,
        "start_col": 1
    },
    "select_table_range": {
        "sheet_name": "Sheet1",
        "start_row": 1,
        "start_col": 1,
        "end_row": -1,
        "end_col": -1
    },
    "set_cell_value": {
        "sheet_name": "Sheet1",
        "row": 1,
        "col": 1,
        "value": None,
        "is_formula": False
    },
    "auto_fill": {
        "sheet_name": "Sheet1",
        "start_row": 1,
        "start_col": 1,
        "end_row": 1,
        "end_col": 1
    },
    "reorder_columns": {
        "sheet_name": "Sheet1",
        "desired_order": []  # Required parameter, no real default
    },
    # Additional tools for PowerPoint (A11Y versions have same defaults as regular versions)
    "set_background_color": {
        "color": "FFFFFF",
        "slide_index": None
    }
}


def normalize_tool_args(function_name: str, args: dict) -> dict:
    """
    Normalize tool arguments by filling in default values for missing parameters.
    
    Args:
        function_name: Name of the tool/function
        args: Arguments provided for the tool
        
    Returns:
        dict: Normalized arguments with all parameters filled in
    """
    if function_name not in TOOL_DEFAULTS:
        # For unknown tools, return args as-is
        return args.copy()
    
    # Start with default values
    normalized_args = TOOL_DEFAULTS[function_name].copy()
    
    # Update with provided values
    normalized_args.update(args)
    
    return normalized_args


def normalize_tool_args_a11y(function_name: str, args: dict) -> dict:
    """
    Normalize A11Y tool arguments by filling in default values for missing parameters.
    Prioritizes control_label over coordinate for A11Y-enabled actions.
    
    Args:
        function_name: Name of the tool/function
        args: Arguments provided for the tool
        
    Returns:
        dict: Normalized arguments with all parameters filled in
    """
    if function_name not in TOOL_DEFAULTS_A11Y:
        # For unknown tools, return args as-is
        return args.copy()
    
    # Start with A11Y default values
    normalized_args = TOOL_DEFAULTS_A11Y[function_name].copy()
    
    # Update with provided values
    normalized_args.update(args)
    
    return normalized_args


def compare_normalized_args(pred_function: str, pred_args: dict, gt_function: str, gt_args: dict) -> bool:
    """
    Compare tool arguments after normalizing both with default values.
    
    Args:
        pred_function: Predicted function name
        pred_args: Predicted arguments
        gt_function: Ground truth function name  
        gt_args: Ground truth arguments
        
    Returns:
        bool: True if normalized arguments match
    """
    if pred_function != gt_function:
        return False
        
    # Normalize both argument sets
    pred_normalized = normalize_tool_args(pred_function, pred_args)
    gt_normalized = normalize_tool_args(gt_function, gt_args)
    
    return pred_normalized == gt_normalized
