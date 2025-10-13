ACTION_PREDICTION_SYS_PROMPT_GPT = """
You are an expert in desktop automation and graphical user interfaces.

You will be provided with a screenshot image of a desktop environment, its resolution, a task instruction, history of previous actions, and a list of supported actions. 

Your objective is to determine the next action that should be taken to accomplish the given instruction given the current state of the desktop environment.

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then, output the next action in JSON format with the keys "thoughts" and "tool_call". Both fields MUST be present.

Please think very hard and carefully about the current state and the instruction before making a decision, and output your reasoning in the "thoughts" field as detailed as possible, like your analysis of the screenshot and past history, the reason of tool call and arguments you would use given your analysis, and your thinking about the status of the task.

The "tool_call" field should contain:
- "function": str, The function/action type to execute
- "args": Dict, The arguments/parameters for the function
- "status": str, The status after performing this action (either "CONTINUE" or "FINISH")

For drag operations, the args should include "start_coordinate" and "end_coordinate":
```json
{
  "thoughts": "The screenshot shows a desktop environment with a Word document open with some text. The user has previously selected a portion of the text and is now trying to drag it to a different location within the document. The estimated start and end coordinates for the drag operation are ...",
  "tool_call": {
    "function": "drag",
    "args": {"start_coordinate": [100, 100], "end_coordinate": [200, 200], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For click operations, the args should include "coordinate":
```json
{
  "thoughts": "The screenshot shows an Excel spreadsheet with a cell selected. The user has previously entered some data into the cell and is now trying to click on the Save button located at the top right corner of the window, which is represented by the coordinates...",
  "tool_call": {
    "function": "click",
    "args": {"coordinate": [150, 30], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For type operations, the args should include "coordinate" and "keys":
```json
{
  "thoughts": "Given the current state of the desktop environment, the user is trying to type 'Hello World' into a text input field. The estimated coordinates for the text input field are xxx. The user wants to clear any existing text before typing the new text.",
  "tool_call": {
    "function": "type",
    "args": {"coordinate": [400, 300], "keys": "Hello World", "clear_current_text": true},
    "status": "CONTINUE"
  }
}
```

If you think the task is finished, output status as "FINISH":
```json
{
  "thoughts": "Given the current state of the desktop environment, and previous actions taken, the task is now complete.",
  "tool_call": {
    "function": "",
    "args": {},
    "status": "FINISH"
  }
}
```

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
Your response string MUST in a pure JSON format. You MUST NOT include any additional text like // # comments or explanations after each field since they will make the JSON invalid. Any comment, thinking and reasoning about the task should be all included in the "thoughts" field.
"""

ACTION_PREDICTION_USER_PROMPT_GPT = """
Instruction: {instruction}
Screenshot resolution: {resolution}

Previous actions performed:
{history}

Supported actions:
{actions}

Current screenshot shows the state after the previous actions. Please analyze the current state and predict the next action to take.

Please provide your reasoning and the next action below:
"""

ACTION_PREDICTION_USER_PROMPT_QWEN = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""

ACTION_PREDICTION_A11Y_USER_PROMPT_QWEN = """You are a helpful assistant with accessibility support. Given a screenshot of the current screen, accessibility (a11y) information with control labels, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

Accessibility Information:
{a11y}

The history of actions are:
{history}

The actions supported are:
{actions}

The screenshot is annotated with numbers corresponding to the control elements in the accessibility information. Each number in the screenshot matches a label in the accessibility list, allowing you to identify and locate specific controls.

**IMPORTANT: When possible, prioritize using control_label over coordinate for actions. Control labels provide more reliable and accessible interaction methods.**

When selecting actions:
- For click/type operations: Use control_label when the target element has a corresponding label in the a11y information
- If control_label is available, set coordinate to null: "control_label": 15, "coordinate": null
- If control_label is not available, use coordinate and set control_label to null: "control_label": null, "coordinate": [100, 200]
- For drag operations: Always use coordinates as they require spatial movement

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state using both visual and accessibility information, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Examples:

For actions with control_label (prioritized):
<tool_call>
{{
  "function": "click",
  "args": {{"control_label": 15, "coordinate": null, "button": "left"}},
  "status": "CONTINUE"
}}
</tool_call>

For actions without control_label (fallback):
<tool_call>
{{
  "function": "click",
  "args": {{"control_label": null, "coordinate": [150, 30], "button": "left"}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot, accessibility information, and previous actions.
"""

SUPPORTED_ACTIONS_WORD = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- insert_table
  - Args:
    - rows: int, The number of rows in the table, starting from 1.
    - columns: int, The number of columns in the table, starting from 1.
  - Example: insert_table(rows=3, columns=3)
- select_text
  - Args:
    - text: str, The exact text to be selected.
  - Example: select_text(text="Hello")
- select_table
  - Args:
    - number: int, The index number of the table to be selected.
  - Example: select_table(number=1)
- select_paragraph
  - Args:
    - start_index: int, The start index of the paragraph to be selected.
    - end_index: int, The end index of the paragraph, if ==-1, select to the end of the document.
    - non_empty: bool, If True, select the non-empty paragraphs only. (Default: True)
  - Example: select_paragraph(start_index=1, end_index=3, non_empty=True)
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pdf". (Default: ".pdf")
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
- set_font
  - Args:
    - font_name: str|None, The name of the font (e.g., "Arial", "Times New Roman", "宋体"). If None, the font name will not be changed. (Default: None)
    - font_size: int|None, The font size (e.g., 12). If None, the font size will not be changed. (Default: None)
  - Example: set_font(font_name="Times New Roman")
</action>"""

SUPPORTED_ACTIONS_EXCEL = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- table2markdown
  - Args:
    - sheet_name: str|int, The name or index of the sheet to get the table content. The index starts from 1.
  - Example: table2markdown(sheet_name=1)
- insert_excel_table
  - Args:
    - table: list[list], The table content to insert. The table is a list of list of strings or numbers.
    - sheet_name: str, The name of the sheet to insert the table.
    - start_row: int, The start row to insert the table, starting from 1.
    - start_col: int, The start column to insert the table, starting from 1.
  - Example: insert_excel_table(table=[["Name", "Age", "Gender"], ["Alice", 30, "Female"], ["Bob", 25, "Male"], ["Charlie", 35, "Male"]], sheet_name="Sheet1", start_row=1, start_col=1)
- select_table_range
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The start row, starting from 1.
    - start_col: int, The start column, starting from 1.
    - end_row: int, The end row. If ==-1, select to the end of the document with content.
    - end_col: int, The end column. If ==-1, select to the end of the document with content.
  - Example: select_table_range(sheet_name="Sheet1", start_row=1, start_col=1, end_row=3, end_col=3)
- set_cell_value
  - Args:
    - sheet_name: str, The name of the sheet.
    - row: int, The row number (1-based).
    - col: int, The column number (1-based).
    - value: str|int|float|None, The value to set in the cell. If None, just select the cell.
    - is_formula: bool, If True, treat the value as a formula, otherwise treat it as a normal value. (Default: False)
  - Example: set_cell_value(sheet_name="Sheet1", row=1, col=1, value="Hello", is_formula=False), set_cell_value(sheet_name="Sheet1", row=2, col=2, value="=SUM(A1:A10)", is_formula=True)
- auto_fill
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The starting row number (1-based).
    - start_col: int, The starting column number (1-based).
    - end_row: int, The ending row number (1-based).
    - end_col: int, The ending column number (1-based).
  - Example: auto_fill(sheet_name="Sheet1", start_row=1, start_col=1, end_row=10, end_col=3)
- reorder_columns
  - Args:
    - sheet_name: str, The name of the sheet.
    - desired_order: list[str], The list of column names in the new order.
  - Example: reorder_columns(sheet_name="Sheet1", desired_order=["Income", "Date", "Expense"]) 
</action>"""

SUPPORTED_ACTIONS_PPT = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- set_background_color
  - Args:
    - color: str, The hex color code (in RGB format) to set the background color.
    - slide_index: list[int]|None, The list of slide indexes to set the background color. If None, set the background color for all slides.
  - Example: set_background_color(color="FFFFFF", slide_index=[1, 2, 3])
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pptx". (Default: ".pptx")
    - current_slide_only: bool, This only applies to ".jpg", ".png", ".gif", ".bmp" and ".tiff" formats. If True, only the current slide will be saved to a PNG file. If False, all slides will be saved into a directory containing multiple PNG files. (Default: False)
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
</action>"""

SUPPORTED_ACTIONS_WORD_NORMAL = """<action>
- click
  - Args:
    - coordinate: [x, y], the normalized position on the screen you want to click at. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[0.1, 0.2], button='left', double=False, pressed=None), click(coordinate=[0.5, 0.8], button='x')
- type
  - Args:
    - coordinate: [x, y], the normalized position on the screen you want to type at. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[0.3, 0.4], keys='Hello'), type(coordinate=[0.6, 0.7], keys='{VK_CONTROL}c'), type(coordinate=[0.8, 0.1], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the normalized position on the screen where the drag starts. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - end_coordinate: [x, y], the normalized position on the screen where the drag ends. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[0.1, 0.2], end_coordinate=[0.8, 0.9], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[0.2, 0.3], end_coordinate=[0.7, 0.6], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the normalized position on the screen to scroll. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[0.5, 0.5], wheel_dist=-5), wheel_mouse_input(coordinate=[0.3, 0.7], wheel_dist=3)
- insert_table
  - Args:
    - rows: int, The number of rows in the table, starting from 1.
    - columns: int, The number of columns in the table, starting from 1.
  - Example: insert_table(rows=3, columns=3)
- select_text
  - Args:
    - text: str, The exact text to be selected.
  - Example: select_text(text="Hello")
- select_table
  - Args:
    - number: int, The index number of the table to be selected.
  - Example: select_table(number=1)
- select_paragraph
  - Args:
    - start_index: int, The start index of the paragraph to be selected.
    - end_index: int, The end index of the paragraph, if ==-1, select to the end of the document.
    - non_empty: bool, If True, select the non-empty paragraphs only. (Default: True)
  - Example: select_paragraph(start_index=1, end_index=3, non_empty=True)
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pdf". (Default: ".pdf")
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
- set_font
  - Args:
    - font_name: str|None, The name of the font (e.g., "Arial", "Times New Roman", "宋体"). If None, the font name will not be changed. (Default: None)
    - font_size: int|None, The font size (e.g., 12). If None, the font size will not be changed. (Default: None)
  - Example: set_font(font_name="Times New Roman")
</action>"""

SUPPORTED_ACTIONS_EXCEL_NORMAL = """<action>
- click
  - Args:
    - coordinate: [x, y], the normalized position on the screen you want to click at. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[0.1, 0.2], button='left', double=False, pressed=None), click(coordinate=[0.5, 0.8], button='x')
- type
  - Args:
    - coordinate: [x, y], the normalized position on the screen you want to type at. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[0.3, 0.4], keys='Hello'), type(coordinate=[0.6, 0.7], keys='{VK_CONTROL}c'), type(coordinate=[0.8, 0.1], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the normalized position on the screen where the drag starts. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - end_coordinate: [x, y], the normalized position on the screen where the drag ends. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[0.1, 0.2], end_coordinate=[0.8, 0.9], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[0.2, 0.3], end_coordinate=[0.7, 0.6], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the normalized position on the screen to scroll. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[0.5, 0.5], wheel_dist=-5), wheel_mouse_input(coordinate=[0.3, 0.7], wheel_dist=3)
- table2markdown
  - Args:
    - sheet_name: str|int, The name or index of the sheet to get the table content. The index starts from 1.
  - Example: table2markdown(sheet_name=1)
- insert_excel_table
  - Args:
    - table: list[list], The table content to insert. The table is a list of list of strings or numbers.
    - sheet_name: str, The name of the sheet to insert the table.
    - start_row: int, The start row to insert the table, starting from 1.
    - start_col: int, The start column to insert the table, starting from 1.
  - Example: insert_excel_table(table=[["Name", "Age", "Gender"], ["Alice", 30, "Female"], ["Bob", 25, "Male"], ["Charlie", 35, "Male"]], sheet_name="Sheet1", start_row=1, start_col=1)
- select_table_range
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The start row, starting from 1.
    - start_col: int, The start column, starting from 1.
    - end_row: int, The end row. If ==-1, select to the end of the document with content.
    - end_col: int, The end column. If ==-1, select to the end of the document with content.
  - Example: select_table_range(sheet_name="Sheet1", start_row=1, start_col=1, end_row=3, end_col=3)
- set_cell_value
  - Args:
    - sheet_name: str, The name of the sheet.
    - row: int, The row number (1-based).
    - col: int, The column number (1-based).
    - value: str|int|float|None, The value to set in the cell. If None, just select the cell.
    - is_formula: bool, If True, treat the value as a formula, otherwise treat it as a normal value. (Default: False)
  - Example: set_cell_value(sheet_name="Sheet1", row=1, col=1, value="Hello", is_formula=False), set_cell_value(sheet_name="Sheet1", row=2, col=2, value="=SUM(A1:A10)", is_formula=True)
- auto_fill
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The starting row number (1-based).
    - start_col: int, The starting column number (1-based).
    - end_row: int, The ending row number (1-based).
    - end_col: int, The ending column number (1-based).
  - Example: auto_fill(sheet_name="Sheet1", start_row=1, start_col=1, end_row=10, end_col=3)
- reorder_columns
  - Args:
    - sheet_name: str, The name of the sheet.
    - desired_order: list[str], The list of column names in the new order.
  - Example: reorder_columns(sheet_name="Sheet1", desired_order=["Income", "Date", "Expense"]) 
</action>"""

SUPPORTED_ACTIONS_PPT_NORMAL = """<action>
- click
  - Args:
    - coordinate: [x, y], the normalized position on the screen you want to click at. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[0.1, 0.2], button='left', double=False, pressed=None), click(coordinate=[0.5, 0.8], button='x')
- type
  - Args:
    - coordinate: [x, y], the normalized position on the screen you want to type at. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True) 
  - Example: type(coordinate=[0.3, 0.4], keys='Hello'), type(coordinate=[0.6, 0.7], keys='{VK_CONTROL}c'), type(coordinate=[0.8, 0.1], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the normalized position on the screen where the drag starts. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - end_coordinate: [x, y], the normalized position on the screen where the drag ends. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[0.1, 0.2], end_coordinate=[0.8, 0.9], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[0.2, 0.3], end_coordinate=[0.7, 0.6], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args: 
    - coordinate: [x, y], the normalized position on the screen to scroll. Values should be in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[0.5, 0.5], wheel_dist=-5), wheel_mouse_input(coordinate=[0.3, 0.7], wheel_dist=3)
- set_background_color
  - Args:
    - color: str, The hex color code (in RGB format) to set the background color.
    - slide_index: list[int]|None, The list of slide indexes to set the background color. If None, set the background color for all slides.
  - Example: set_background_color(color="FFFFFF", slide_index=[1, 2, 3])
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pptx". (Default: ".pptx")
    - current_slide_only: bool, This only applies to ".jpg", ".png", ".gif", ".bmp" and ".tiff" formats. If True, only the current slide will be saved to a PNG file. If False, all slides will be saved into a directory containing multiple PNG files. (Default: False)
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
</action>"""


SYSTEM_PROMPT_ACTION_PREDICTION_WITH_A11Y = """<a11y>
{a11y}
</a11y>

You are a helpful assistant. Given a screenshot and a11y information of the current screen and control labels information(a11y), user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{historys}

The actions supported are:
{actions}

Ouput your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>
"""

ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT = """
You are an expert in desktop automation and graphical user interfaces with accessibility support.


You will be provided with the following inputs:

1. **Screenshot**: An image of a desktop environment.
2. **Accessibility (a11y) information**: This includes a list of control element labels and the textual name of the currently active application.
3. **Task instruction**: A description of the action or goal to be completed.
4. **History of previous actions**: A sequential log of actions already performed.
5. **Supported actions**: A list of all actions that can be performed in this environment.

The screenshot is annotated with numbers corresponding to the control elements in the accessibility information. Each number in the screenshot matches a label in the accessibility list, allowing you to identify and locate specific controls.


Your objective is to determine the next action that should be taken to accomplish the given instruction given the current state of the desktop environment and the available accessibility information.

Use all the provided information—including the screenshot, accessibility data, task instructions, action history, and supported actions—to reason about the next appropriate action accurately.


**IMPORTANT: When possible, prioritize using control_label over coordinate for actions. Control labels provide more reliable and accessible interaction methods.**

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state using both visual and accessibility information, and determine what action should be taken next based on the instruction and previous actions.

Then, output the next action in JSON format with the keys "thoughts" and "tool_call". Both fields MUST be present.

Please think very hard and carefully about the current state and the instruction before making a decision, and output your reasoning in the "thoughts" field as detailed as possible, including:
- Your analysis of the screenshot and accessibility information
- How you identified the target control using control labels or visual elements  
- The reason for your tool call and arguments selection
- Your thinking about the status of the task

The "tool_call" field should contain:
- "function": str, The function/action type to execute
- "args": Dict, The arguments/parameters for the function
- "status": str, The status after performing this action (either "CONTINUE" or "FINISH")

For click operations, prioritize control_label over coordinate:
```json
{
  "thoughts": "The screenshot shows an Excel spreadsheet. I can see from the a11y information that there is a Save button with control_label=15. I should use the control_label for more reliable interaction rather than trying to estimate coordinates.",
  "tool_call": {
    "function": "click",
    "args": {"control_label": 15, "coordinate": null, "button": "left"},
    "status": "CONTINUE"
  }
}
```

If control_label is not available, fall back to coordinate:
```json
{
  "thoughts": "I need to click on a specific area that doesn't have a control_label in the a11y information. Based on the screenshot analysis, I can estimate the coordinates for this location.",
  "tool_call": {
    "function": "click",
    "args": {"control_label": null, "coordinate": [150, 30], "button": "left"},
    "status": "CONTINUE"
  }
}
```

For type operations, prioritize control_label over coordinate:
```json
{
  "thoughts": "I need to type text into an input field. From the a11y information, I can see there's a text input control with control_label=8. Using the control_label ensures I'm targeting the correct input field.",
  "tool_call": {
    "function": "type",
    "args": {"control_label": 8, "coordinate": null, "keys": "Hello World", "clear_current_text": true},
    "status": "CONTINUE"
  }
}
```

For drag operations, coordinates are still required:
```json
{
  "thoughts": "I need to perform a drag operation from one location to another. Drag operations require specific start and end coordinates as they involve spatial movement that cannot be represented by control labels alone.",
  "tool_call": {
    "function": "drag",
    "args": {"start_coordinate": [100, 100], "end_coordinate": [200, 200], "button": "left"},
    "status": "CONTINUE"
  }
}
```

If you think the task is finished, output status as "FINISH":
```json
{
  "thoughts": "Given the current state of the desktop environment, accessibility information, and previous actions taken, the task is now complete.",
  "tool_call": {
    "function": "",
    "args": {},
    "status": "FINISH"
  }
}
```

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot, accessibility information, and previous actions.

Your response string MUST be in pure JSON format. You MUST NOT include any additional text like // # comments or explanations after each field since they will make the JSON invalid. Any comment, thinking and reasoning about the task should be all included in the "thoughts" field.
"""

ACTION_PREDICTION_A11Y_USER_PROMPT_GPT = """
Instruction: {instruction}

Accessibility Information:
{a11y}

Previous actions performed:
{history}

Supported actions:
{actions}

Current screenshot shows the state after the previous actions. Please analyze the current state using both visual information and accessibility data to predict the next action to take.

Please provide your reasoning and the next action below in JSON format without any additional text.
"""



# A11y versions of supported actions that use control_label instead of coordinate
SUPPORTED_ACTIONS_WORD_A11Y = """<action>
- click
  - Args:
      - control_label: int | None
        Description: The label of the control to click. You should prioritize using `control_label` whenever possible. Only use `None` if the target location does not have a corresponding control label. The label must exist in the provided accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to click, used only when the target is not listed in the a11y information and has no control label. In this case, set `control_label` to `None` and estimate the coordinate.
      - button: str
        Description: The mouse button to click. Options are `'left'`, `'right'`, `'middle'`, or `'x'`. Default is `'left'`.
      - double: bool
        Description: Whether to perform a double click. Default is `False`.
      - pressed: str | None
        Description: The keyboard key to hold while clicking. Common keys include `'CONTROL'` (Ctrl), `'SHIFT'` (Shift), `'MENU'` (Alt), etc. Do not include prefixes like `VK_` or braces. For example, use `'CONTROL'` for the Control key. Default is `None`.
  - Example: click(control_label=13, coordinate=None, button='left', double=False, pressed=None), click(control_label=None, coordinate=[100, 100], button='x')
- type
  - Args:
      - control_label: int | None
        Description: The label of the control to type into. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to type at, used only when the target control is not listed in the a11y information. Set `control_label` to `None` and use this coordinate.
      - keys: str
        Description: The keys to type. Can include any keyboard key, with special keys represented by their virtual key codes (e.g., `"{VK_CONTROL}c"` for Ctrl+C).
      - clear_current_text: bool
        Description: Whether to clear the current text in the control before typing new text. Default is `False`.
      - control_focus: bool
        Description: Whether to focus the selected control before typing. If `False`, hotkeys will be sent to the application window. Default is `True`.
  - Example:
      - type(coordinate=[100, 100], keys='Hello')
      - type(coordinate=[100, 100], keys='{VK_CONTROL}c')
      - type(coordinate=[100, 100], keys='{TAB 2}')
- drag
  - Args:
      - start_coordinate: [x, y]
        Description: The absolute screen position where the drag starts.
      - end_coordinate: [x, y]
        Description: The absolute screen position where the drag ends.
      - button: str
        Description: The mouse button to use for dragging. Options: `'left'` or `'right'`. Default is `'left'`.
      - duration: float
        Description: Duration of the drag action in seconds. Default is `1.0`.
      - key_hold: str | None
        Description: Keyboard key to hold while dragging. Common keys include `'shift'`, `'control'`, `'alt'`, etc. Use lowercase names. Default is `None`.
  - Example:
      - drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None)
      - drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')

- wheel_mouse_input
  - Args:
      - control_label: int | None
        Description: The label of the control for wheel input. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the a11y information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to scroll, used only when the target control is not in the a11y information. Set `control_label` to `None` and use this coordinate.
      - wheel_dist: int
        Description: Number of wheel notches to scroll. Positive values indicate upward scrolling; negative values indicate downward scrolling.
  - Example:
      - wheel_mouse_input(control_label=13, coordinate=None, wheel_dist=-5)
      - wheel_mouse_input(control_label=None, coordinate=[100, 100], wheel_dist=3)

- insert_table
  - Args:
      - rows: int
        Description: The number of rows in the table.
      - columns: int
        Description: The number of columns in the table.
  - Example:
      - insert_table(rows=3, columns=3)

- select_text
  - Args:
      - text: str
        Description: The exact text to be selected in the document.
  - Example:
      - select_text(text="Hello")

- select_table
  - Args:
      - number: int
        Description: The index number of the table to be selected (1-based indexing).
  - Example:
      - select_table(number=1)

- select_paragraph
  - Args:
      - start_index: int
        Description: The start index of the paragraph to select (1-based indexing).
      - end_index: int
        Description: The end index of the paragraph. If set to -1, select until the end of the document.
      - non_empty: bool
        Description: If True, only select non-empty paragraphs. Default is True.
  - Example:
      - select_paragraph(start_index=1, end_index=3, non_empty=True)

- save_as
  - Args:
      - file_dir: str
        Description: Directory to save the file. Defaults to the current directory if not specified. Default is `""`.
      - file_name: str
        Description: Name of the file without extension. Defaults to the current document's name if not specified. Default is `""`.
      - file_ext: str
        Description: File extension. Defaults to `.pdf` if not specified. Default is `.pdf`.
  - Example:
      - save_as(file_dir="", file_name="", file_ext=".pdf")

- set_font
  - Args:
      - font_name: str | None
        Description: The name of the font (e.g., `"Arial"`, `"Times New Roman"`, `"宋体"`). If `None`, the font name will not be changed. Default is `None`.
      - font_size: int | None
        Description: The font size (e.g., `12`). If `None`, the font size will not be changed. Default is `None`.
  - Example:
      - set_font(font_name="Times New Roman")

</action>"""

SUPPORTED_ACTIONS_EXCEL_A11Y = """<action>
- click
  - Args:
      - control_label: int | None
        Description: The label of the control to click. You should prioritize using `control_label` whenever possible. Only use `None` if the target location does not have a corresponding control label. The label must exist in the provided accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to click, used only when the target is not listed in the a11y information and has no control label. In this case, set `control_label` to `None` and estimate the coordinate.
      - button: str
        Description: The mouse button to click. Options are `'left'`, `'right'`, `'middle'`, or `'x'`. Default is `'left'`.
      - double: bool
        Description: Whether to perform a double click. Default is `False`.
      - pressed: str | None
        Description: The keyboard key to hold while clicking. Common keys include `'CONTROL'` (Ctrl), `'SHIFT'` (Shift), `'MENU'` (Alt), etc. Do not include prefixes like `VK_` or braces. For example, use `'CONTROL'` for the Control key. Default is `None`.
  - Example: click(control_label=13, coordinate=None, button='left', double=False, pressed=None), click(control_label=None, coordinate=[100, 100], button='x')
- type
  - Args:
      - control_label: int | None
        Description: The label of the control to type into. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to type at, used only when the target control is not listed in the a11y information. Set `control_label` to `None` and use this coordinate.
      - keys: str
        Description: The keys to type. Can include any keyboard key, with special keys represented by their virtual key codes (e.g., `"{VK_CONTROL}c"` for Ctrl+C).
      - clear_current_text: bool
        Description: Whether to clear the current text in the control before typing new text. Default is `False`.
      - control_focus: bool
        Description: Whether to focus the selected control before typing. If `False`, hotkeys will be sent to the application window. Default is `True`.
  - Example:
      - type(coordinate=[100, 100], keys='Hello')
      - type(coordinate=[100, 100], keys='{VK_CONTROL}c')
      - type(coordinate=[100, 100], keys='{TAB 2}')
- drag
  - Args:
      - start_coordinate: [x, y]
        Description: The absolute screen position where the drag starts.
      - end_coordinate: [x, y]
        Description: The absolute screen position where the drag ends.
      - button: str
        Description: The mouse button to use for dragging. Options: `'left'` or `'right'`. Default is `'left'`.
      - duration: float
        Description: Duration of the drag action in seconds. Default is `1.0`.
      - key_hold: str | None
        Description: Keyboard key to hold while dragging. Common keys include `'shift'`, `'control'`, `'alt'`, etc. Use lowercase names. Default is `None`.
  - Example:
      - drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None)
      - drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')

- wheel_mouse_input
  - Args:
      - control_label: int | None
        Description: The label of the control for wheel input. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the a11y information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to scroll, used only when the target control is not in the a11y information. Set `control_label` to `None` and use this coordinate.
      - wheel_dist: int
        Description: Number of wheel notches to scroll. Positive values indicate upward scrolling; negative values indicate downward scrolling.
  - Example:
      - wheel_mouse_input(control_label=13, coordinate=None, wheel_dist=-5)
      - wheel_mouse_input(control_label=None, coordinate=[100, 100], wheel_dist=3)
- table2markdown
  - Args:
      - sheet_name: str | int
        Description: The name or 1-based index of the sheet from which to extract the table content.
  - Example:
      - table2markdown(sheet_name=1)

- insert_excel_table
  - Args:
      - table: list[list[str | int | float]]
        Description: The table content to insert. Represented as a list of rows, each row being a list of strings or numbers.
      - sheet_name: str
        Description: The name of the sheet to insert the table into.
      - start_row: int
        Description: The starting row to insert the table, 1-based indexing.
      - start_col: int
        Description: The starting column to insert the table, 1-based indexing.
  - Example:
      - insert_excel_table(
          table=[["Name", "Age", "Gender"],
                 ["Alice", 30, "Female"],
                 ["Bob", 25, "Male"],
                 ["Charlie", 35, "Male"]],
          sheet_name="Sheet1",
          start_row=1,
          start_col=1
        )

- select_table_range
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - start_row: int
        Description: The start row for selection, 1-based indexing.
      - start_col: int
        Description: The start column for selection, 1-based indexing.
      - end_row: int
        Description: The end row for selection. If set to -1, select until the last row with content.
      - end_col: int
        Description: The end column for selection. If set to -1, select until the last column with content.
  - Example:
      - select_table_range(sheet_name="Sheet1", start_row=1, start_col=1, end_row=3, end_col=3)

- set_cell_value
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - row: int
        Description: The row number of the cell, 1-based indexing.
      - col: int
        Description: The column number of the cell, 1-based indexing.
      - value: str | int | float | None
        Description: The value to set in the cell. If `None`, only select the cell without changing its content.
      - is_formula: bool
        Description: If True, treat `value` as a formula; otherwise treat it as a normal value. Default is `False`.
  - Example:
      - set_cell_value(sheet_name="Sheet1", row=1, col=1, value="Hello", is_formula=False)
      - set_cell_value(sheet_name="Sheet1", row=2, col=2, value="=SUM(A1:A10)", is_formula=True)

- auto_fill
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - start_row: int
        Description: The starting row of the auto-fill range, 1-based indexing.
      - start_col: int
        Description: The starting column of the auto-fill range, 1-based indexing.
      - end_row: int
        Description: The ending row of the auto-fill range, 1-based indexing.
      - end_col: int
        Description: The ending column of the auto-fill range, 1-based indexing.
  - Example:
      - auto_fill(sheet_name="Sheet1", start_row=1, start_col=1, end_row=10, end_col=3)

- reorder_columns
  - Args:
      - sheet_name: str
        Description: The name of the sheet.
      - desired_order: list[str]
        Description: A list of column names specifying the new order.
  - Example:
      - reorder_columns(sheet_name="Sheet1", desired_order=["Income", "Date", "Expense"])

</action>"""

SUPPORTED_ACTIONS_PPT_A11Y = """<action>
- click
  - Args:
      - control_label: int | None
        Description: The label of the control to click. You should prioritize using `control_label` whenever possible. Only use `None` if the target location does not have a corresponding control label. The label must exist in the provided accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to click, used only when the target is not listed in the a11y information and has no control label. In this case, set `control_label` to `None` and estimate the coordinate.
      - button: str
        Description: The mouse button to click. Options are `'left'`, `'right'`, `'middle'`, or `'x'`. Default is `'left'`.
      - double: bool
        Description: Whether to perform a double click. Default is `False`.
      - pressed: str | None
        Description: The keyboard key to hold while clicking. Common keys include `'CONTROL'` (Ctrl), `'SHIFT'` (Shift), `'MENU'` (Alt), etc. Do not include prefixes like `VK_` or braces. For example, use `'CONTROL'` for the Control key. Default is `None`.
  - Example: click(control_label=13, coordinate=None, button='left', double=False, pressed=None), click(control_label=None, coordinate=[100, 100], button='x')
- type
  - Args:
      - control_label: int | None
        Description: The label of the control to type into. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the accessibility (a11y) information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to type at, used only when the target control is not listed in the a11y information. Set `control_label` to `None` and use this coordinate.
      - keys: str
        Description: The keys to type. Can include any keyboard key, with special keys represented by their virtual key codes (e.g., `"{VK_CONTROL}c"` for Ctrl+C).
      - clear_current_text: bool
        Description: Whether to clear the current text in the control before typing new text. Default is `False`.
      - control_focus: bool
        Description: Whether to focus the selected control before typing. If `False`, hotkeys will be sent to the application window. Default is `True`.
  - Example:
      - type(coordinate=[100, 100], keys='Hello')
      - type(coordinate=[100, 100], keys='{VK_CONTROL}c')
      - type(coordinate=[100, 100], keys='{TAB 2}')
- drag
  - Args:
      - start_coordinate: [x, y]
        Description: The absolute screen position where the drag starts.
      - end_coordinate: [x, y]
        Description: The absolute screen position where the drag ends.
      - button: str
        Description: The mouse button to use for dragging. Options: `'left'` or `'right'`. Default is `'left'`.
      - duration: float
        Description: Duration of the drag action in seconds. Default is `1.0`.
      - key_hold: str | None
        Description: Keyboard key to hold while dragging. Common keys include `'shift'`, `'control'`, `'alt'`, etc. Use lowercase names. Default is `None`.
  - Example:
      - drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None)
      - drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')

- wheel_mouse_input
  - Args:
      - control_label: int | None
        Description: The label of the control for wheel input. Prioritize using `control_label` if available. Only use `None` if the target does not have a corresponding label in the a11y information.
      - coordinate: [x, y] (Optional)
        Description: The absolute screen position to scroll, used only when the target control is not in the a11y information. Set `control_label` to `None` and use this coordinate.
      - wheel_dist: int
        Description: Number of wheel notches to scroll. Positive values indicate upward scrolling; negative values indicate downward scrolling.
  - Example:
      - wheel_mouse_input(control_label=13, coordinate=None, wheel_dist=-5)
      - wheel_mouse_input(control_label=None, coordinate=[100, 100], wheel_dist=3)
- set_background_color
  - Args:
      - color: str
        Description: The background color to set, specified as a hex RGB code (e.g., "FFFFFF" for white).
      - slide_index: list[int] | None
        Description: A list of slide indexes to apply the background color to. If `None`, the color will be applied to all slides.
  - Example:
      - set_background_color(color="FFFFFF", slide_index=[1, 2, 3])

- save_as
  - Args:
      - file_dir: str
        Description: Directory to save the file. Defaults to the current directory if not specified. Default is `""`.
      - file_name: str
        Description: Name of the file without extension. Defaults to the current presentation's name if not specified. Default is `""`.
      - file_ext: str
        Description: File extension. Defaults to `.pptx` if not specified. Default is `.pptx`.
      - current_slide_only: bool
        Description: Applies only when saving as `.jpg`, `.png`, `.gif`, `.bmp`, or `.tiff`. If `True`, only the current slide will be saved; if `False`, all slides will be saved as separate files in a directory. Default is `False`.
  - Example:
      - save_as(file_dir="", file_name="", file_ext=".pdf")

</action>"""
