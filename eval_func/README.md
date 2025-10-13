# Eval Function
Code in `tool_func.py`
```python
def eval_tool(predict: dict | str, ground_truth: dict | str, ground_bbox: list | dict | None = None) -> Tuple[int, int, int]:
    """
    Evaluate the tool prediction against the ground truth.

    Args:
        predict: The predicted tool call.
        ground_truth: The ground truth tool call.
        ground_bbox: The ground truth bounding box.

    Returns:
        A tuple of (function_match, args_match, status_match). For example: (1, 1, 1)
    """
```

# Data Format

```json
[
  {
    "id": "excel_bing_search_excel_4s_1_2", // unique id
    "images": [
      "images/excel_4s_1/action_step4.png" // image path
    ],
    "conversation": [
      {
        "from": "human",
        "value": "<image>\nYou are a helpful assistant ..." // input prompt
      },
      {
        "from": "gpt",
        "value": "<tool_call>\n{\n    \"function\": \"type\",\n    \"args\": {\n        \"text\": \"=ATAN(1)\",\n        \"clear_current_text\": true,\n        \"coordinate\": [627.6, 238.9]\n    },\n    \"status\": \"CONTINUE\"\n}\n</tool_call>" // ground truth action
      },
    ],
    "reward": 0, // true: 1; false: 0
    "bbox": [0,0,0,0] // bbox or None
  }   
]          
```

# Prompt Template
``` python
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
```