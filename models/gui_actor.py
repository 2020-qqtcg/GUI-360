import ast
import base64
import json
import re
import numpy as np
from io import BytesIO
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image

from models.base import BaseModel
from prompts.prompt_action_prediction import ACTION_PREDICTION_USER_PROMPT_QWEN

# Try to import qwen-vl-utils for smart_resize, fall back to manual implementation if not available
try:
    from qwen_vl_utils import smart_resize
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
        """
        Manual implementation of smart_resize function.
        Adjusts image dimensions to be multiples of factor and within pixel range.
        """
        # Calculate adjusted dimensions
        new_height = int(np.round(height / factor) * factor)
        new_width = int(np.round(width / factor) * factor)
        new_pixels = new_height * new_width

        # If total pixels are out of range, further adjust dimensions
        if new_pixels < min_pixels:
            scale = np.sqrt(min_pixels / new_pixels)
            new_height = int(np.round(new_height * scale / factor) * factor)
            new_width = int(np.round(new_width * scale / factor) * factor)
        elif new_pixels > max_pixels:
            scale = np.sqrt(max_pixels / new_pixels)
            new_height = int(np.round(new_height * scale / factor) * factor)
            new_width = int(np.round(new_width * scale / factor) * factor)

        return new_height, new_width


class GuiActor(BaseModel):
    """
    GUI Actor model class for interacting with remote GUI agent services.
    Adapted from CustomAgent code to fit the evaluation framework interface requirements.
    """
    
    _MAX_STEP = 50

    def __init__(self, api_url: str = "http://20.91.255.106:19806/evaluate", model_name: str = "gui_actor", use_smart_resize: bool = True):
        """
        Initialize GUI Actor model.
        
        Args:
            api_url: API endpoint for GUI Actor service
            model_name: Model name
            use_smart_resize: Whether to use smart_resize for better coordinate accuracy
        """
        super().__init__(model_name)
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}
        self.previous_actions = []
        self.step_counter = 0
        self.use_smart_resize = use_smart_resize
        
        # Store original image dimensions for coordinate transformation
        self.original_dimensions = None
        self.resized_dimensions = None
        
        print(f"Initialized GUI Actor model")
        print(f"API URL: {self.api_url}")
        print(f"Model: {self.model_name}")
        print(f"Smart resize enabled: {self.use_smart_resize}")
        if not HAS_QWEN_VL_UTILS and self.use_smart_resize:
            print("Warning: qwen-vl-utils not found, using manual smart_resize implementation")

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
        except Exception as e:
            print(f"Error encoding image: {e}")
            raise

    def _preprocess_image_with_smart_resize(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image using smart_resize for better coordinate accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Image.Image: Processed image
        """
        if not self.use_smart_resize:
            return image
            
        try:
            # Store original dimensions
            self.original_dimensions = image.size  # (width, height)
            original_width, original_height = self.original_dimensions
            
            # Apply smart_resize to get optimal dimensions
            resized_height, resized_width = smart_resize(original_height, original_width)
            self.resized_dimensions = (resized_width, resized_height)
            
            # Resize image if dimensions changed
            if (resized_width, resized_height) != (original_width, original_height):
                resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
                print(f"Smart resize applied: {original_width}x{original_height} -> {resized_width}x{resized_height}")
                return resized_image
            else:
                print(f"No resize needed: {original_width}x{original_height}")
                return image
                
        except Exception as e:
            print(f"Error in smart resize preprocessing: {e}")
            # Fall back to original image
            self.original_dimensions = image.size
            self.resized_dimensions = image.size
            return image

    def _transform_coordinates_to_original(self, coordinates: List[float]) -> List[float]:
        """
        Transform coordinates from resized image back to original image coordinates.
        
        Args:
            coordinates: [x, y] coordinates from model output
            
        Returns:
            List[float]: Transformed coordinates for original image
        """
        if not self.use_smart_resize or not self.original_dimensions or not self.resized_dimensions:
            return coordinates
            
        if len(coordinates) < 2:
            return coordinates
            
        try:
            original_width, original_height = self.original_dimensions
            resized_width, resized_height = self.resized_dimensions
            
            # Calculate scaling factors
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height
            
            # Transform coordinates
            transformed_x = coordinates[0] * scale_x
            transformed_y = coordinates[1] * scale_y
            
            # Handle additional coordinates if present (e.g., for rectangles)
            transformed_coords = [transformed_x, transformed_y]
            if len(coordinates) > 2:
                for i in range(2, len(coordinates)):
                    if i % 2 == 0:  # x coordinate
                        transformed_coords.append(coordinates[i] * scale_x)
                    else:  # y coordinate
                        transformed_coords.append(coordinates[i] * scale_y)
            
            print(f"Coordinate transformation: {coordinates} -> {transformed_coords}")
            print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
            return transformed_coords
            
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return coordinates

    def predict(self, system_prompt: str, user_prompt: str, image_path: str, max_tokens: int = 2048, temperature: float = 0.0, *args, **kwargs) -> str:
        """
        Universal prediction method that accepts system and user prompts.
        
        Args:
            system_prompt: System prompt (may not be used for GUI Actor)
            user_prompt: User prompt containing task instructions
            image_path: Path to image file
            max_tokens: Maximum number of tokens (unused)
            temperature: Temperature parameter (unused)
            
        Returns:
            str: Model response text
        """
        try:
            # Extract task instruction from user prompt
            # Simplified handling, assuming user_prompt contains complete task description
            task = user_prompt
            
            # Load image
            if isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
                # If it's a URL, download the image
                response = requests.get(image_path)
                screenshot = Image.open(BytesIO(response.content))
            else:
                screenshot = Image.open(image_path)
            
            # Apply smart resize preprocessing
            processed_screenshot = self._preprocess_image_with_smart_resize(screenshot)
            
            # Build request data
            prompt_data = self.prompt_construction(task, processed_screenshot)
            
            # Call inference interface
            results = self.inference(prompt_data)
            
            if not results:
                return "Error: Inference failed"
                
            # Return prediction text
            return results
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"Error: {str(e)}"

    def construct_grounding_prompt(self, thought: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for grounding task.
        
        Args:
            thought: Thought content
            resolution: Screen resolution
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # GUI Actor is mainly used for action prediction, grounding functionality may be limited
        system_prompt = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
        user_prompt = f"Please locate the element related to: {thought}"
        return system_prompt, user_prompt

    def construct_action_prompt(self, instruction: str, history: str, actions: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for action prediction task.
        
        Args:
            instruction: Instruction
            history: History records
            actions: Supported actions
            resolution: Screen resolution
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = ""
        
        # Format history records
        history_formatted = "\n".join(history) if isinstance(history, list) else history if history else ""
        
        user_prompt = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
            instruction=instruction,
            history=history_formatted,
            actions=actions
        )
        return system_prompt, user_prompt

    def prompt_construction(self, task: str, screenshot: Image) -> Dict[str, Any]:
        """
        Build prompt data to send to GUI Actor service.
        
        Args:
            task: Task description
            screenshot: Screenshot PIL image
            
        Returns:
            Dict: Request data
        """
        buffer = BytesIO()
        try:
            screenshot.save(buffer, format="PNG")
        except Exception as e:
            print(f"Error saving screenshot to buffer: {e}")
            byte_data = b""
        else:
            byte_data = buffer.getvalue()

        encoded_string = base64.b64encode(byte_data).decode("utf-8")

        data = {
            "system_message": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
            "image": encoded_string,
            "previous_actions": self.previous_actions,
            "user_instruction": task,
        }
        return data

    def inference(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference request.
        
        Args:
            prompt: Request data
            
        Returns:
            Dict: Response data
        """
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.api_url, headers=self.headers, data=json.dumps(prompt), timeout=20
                )
                response.raise_for_status()
                results = response.json()
                print("Response from GUI Actor server:")
                print(json.dumps(results, indent=4))
                return results
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code in [500, 502, 503, 504]:
                    retry_count += 1
                    print(f"Retrying... ({retry_count}/{max_retries})")
                else:
                    return {}
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                retry_count += 1
                print(f"Retrying... ({retry_count}/{max_retries})")
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON response: {e}")
                print(f"Response text: {response.text if 'response' in locals() else 'No response object'}")
                return {}

        print("Max retries reached. Unable to get a valid response.")
        return {}

    def parse_coordinates(self, response) -> list[float]:
        """
        Parse the coordinates from the model response and transform them to original image coordinates.
        """
        if isinstance(response, dict):
            # Handle structured response from API
            top_points = response.get("topk_points", [])[0] if response.get("topk_points") else [0.0, 0.0]
            width, height = response.get("image_size", {}).get("width", 0), response.get("image_size", {}).get("height", 0)
            print(f"Width: {width}, Height: {height}")
            print(f"Top points: {top_points}")
            coordinates = [top_points[0] * width, top_points[1] * height]
        else:
            # Handle text response (fallback)
            coordinates = [0.0, 0.0]
            print(f"Warning: Unexpected response format for coordinate parsing: {type(response)}")
        
        # Transform coordinates back to original image dimensions
        transformed_coordinates = self._transform_coordinates_to_original(coordinates)
        return transformed_coordinates

    def parse_action(self, response: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Parse action information from model response.
        
        Args:
            response: Model response text
            
        Returns:
            Tuple: (function_name, args_dict, status) or (None, None, None) if parsing fails
        """
        try:
            # First try to get structured data from complete response
            if isinstance(response, str):
                try:
                    # If response is JSON string, try to parse it
                    response_data = json.loads(response)
                except json.JSONDecodeError:
                    # If not JSON, assume it contains prediction text
                    response_data = {"pred_text": response}
            else:
                response_data = response

            pred_text = response_data.get("pred_text", response if isinstance(response, str) else "")
            
            if not pred_text:
                return None, None, None

            # Parse natural language action
            natural_language_action = self.parse_natural_language_action(pred_text)
            
            # Parse pyautogui calls
            pyautogui_actions_dicts = self.parse_pyautogui_calls(pred_text)
            
            if not pyautogui_actions_dicts:
                # If no pyautogui actions parsed, try to parse as framework format
                return self._parse_framework_action(pred_text)
            
            # Convert first pyautogui action to framework format
            if pyautogui_actions_dicts:
                action_dict = pyautogui_actions_dicts[0]
                function_name = action_dict.get("function", "")
                
                # Map pyautogui functions to framework functions
                function_mapping = {
                    "click": "click",
                    "doubleClick": "click",
                    "rightClick": "click", 
                    "type": "type",
                    "drag": "drag",
                    "dragTo": "drag",
                    "scroll": "wheel_mouse_input",
                    "hotkey": "type"
                }
                
                mapped_function = function_mapping.get(function_name, function_name)
                
                # Build arguments
                args_dict = {}
                kw_args = action_dict.get("args", {})
                pos_args = action_dict.get("*args", [])
                
                if mapped_function == "click":
                    if "x" in kw_args and "y" in kw_args:
                        args_dict["coordinate"] = [kw_args["x"], kw_args["y"]]
                    elif len(pos_args) >= 2:
                        args_dict["coordinate"] = [pos_args[0], pos_args[1]]
                    
                    if function_name == "doubleClick":
                        args_dict["double"] = True
                    elif function_name == "rightClick":
                        args_dict["button"] = "right"
                        
                elif mapped_function == "type":
                    if "x" in kw_args and "y" in kw_args:
                        args_dict["coordinate"] = [kw_args["x"], kw_args["y"]]
                    elif len(pos_args) >= 2:
                        args_dict["coordinate"] = [pos_args[0], pos_args[1]]
                    
                    # Get text to input
                    if pos_args:
                        args_dict["keys"] = str(pos_args[-1])  # Usually text is the last parameter
                        
                elif mapped_function == "drag":
                    if "x" in kw_args and "y" in kw_args:
                        args_dict["start_coordinate"] = [kw_args["x"], kw_args["y"]]
                    elif len(pos_args) >= 2:
                        args_dict["start_coordinate"] = [pos_args[0], pos_args[1]]
                    
                    if len(pos_args) >= 4:
                        args_dict["end_coordinate"] = [pos_args[2], pos_args[3]]
                        
                elif mapped_function == "wheel_mouse_input":
                    if "x" in kw_args and "y" in kw_args:
                        args_dict["coordinate"] = [kw_args["x"], kw_args["y"]]
                    elif len(pos_args) >= 2:
                        args_dict["coordinate"] = [pos_args[0], pos_args[1]]
                    
                    if "clicks" in kw_args:
                        args_dict["wheel_dist"] = kw_args["clicks"]
                    elif len(pos_args) >= 3:
                        args_dict["wheel_dist"] = pos_args[2]
                
                return mapped_function, args_dict, "CONTINUE"
            
            return None, None, None
            
        except Exception as e:
            print(f"Error parsing action from response: {e}")
            return None, None, None

    def parse_thoughts(self, response) -> str:
        """
        Parse the reasoning thoughts from the model response.
        """
        return ""

    def _parse_framework_action(self, text: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Parse framework format action (tool_call format).
        
        Args:
            text: Response text
            
        Returns:
            Tuple: (function_name, args_dict, status)
        """
        # Try to parse tool_call format
        pattern = r'<tool_call>\s*\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}\s*</tool_call>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            function_name = match.group(1)
            args_str = match.group(2)
            status = match.group(3)
            
            try:
                args_dict = json.loads(args_str)
                return function_name, args_dict, status
            except json.JSONDecodeError:
                return function_name, {}, status
        
        return None, None, None

    @staticmethod
    def parse_natural_language_action(text: str) -> str:
        """
        Parse natural language action description.
        
        Args:
            text: Response text
            
        Returns:
            str: Natural language action description
        """
        for line in text.splitlines():
            if line.strip().lower().startswith("action:"):
                return line.partition(":")[2].strip()
        return ""

    @staticmethod
    def parse_pyautogui_calls(input_str: str) -> List[Dict[str, Any]]:
        """
        Parse pyautogui function calls.
        
        Args:
            input_str: Input string
            
        Returns:
            List[Dict]: List of parsed actions
        """
        pattern = r"pyautogui\.(\w+)\((.*?)\)"
        matches = re.findall(pattern, input_str)
        parsed_actions = []

        for func_name, args_str in matches:
            keyword_args = {}
            positional_args_list = []

            if args_str.strip():
                try:
                    tree = ast.parse(f"dummy_call_name({args_str})", mode='eval')
                    call_node = tree.body

                    for arg_node in call_node.args:
                        positional_args_list.append(ast.literal_eval(arg_node))

                    for kw_node in call_node.keywords:
                        keyword_args[kw_node.arg] = ast.literal_eval(kw_node.value)

                except (SyntaxError, ValueError, TypeError) as e:
                    print(f"Argument parsing failed for: pyautogui.{func_name}({args_str}) -> {e}")
                except Exception as e:
                    print(f"Unknown argument parsing error for: pyautogui.{func_name}({args_str}) -> {e}")

            parsed_actions.append({
                "function": func_name,
                "args": keyword_args,
                "*args": positional_args_list
            })
        return parsed_actions

    def reset(self):
        """Reset agent state."""
        self.previous_actions = []
        self.step_counter = 0
        # Reset image dimensions state
        self.original_dimensions = None
        self.resized_dimensions = None

    def run(self, task: str, initial_screenshot: Image) -> None:
        """
        Run task (for compatibility with original CustomAgent interface).
        
        Args:
            task: Task description
            initial_screenshot: Initial screenshot
        """
        self.reset()
        current_screenshot = initial_screenshot

        while self.step_counter < self._MAX_STEP:
            natural_language_action, pyautogui_actions = self.step(task, current_screenshot)

            if not pyautogui_actions:
                print("No actions returned, stopping run.")
                break

            print(f"Run - Step {self.step_counter}: NLA: {natural_language_action}, Actions: {pyautogui_actions}")

            if natural_language_action:
                self.previous_actions.append(natural_language_action)

            self.step_counter += 1
            if self.step_counter >= self._MAX_STEP:
                print("Max steps reached in run.")
                break
            
            print("Stopping run after one iteration for safety as execute_action and screenshot updates are not implemented.")
            break

    def step(self, task: str, current_screenshot: Image) -> Tuple[str, List[str]]:
        """
        Execute one step (for compatibility with original CustomAgent interface).
        
        Args:
            task: Task description
            current_screenshot: Current screenshot
            
        Returns:
            Tuple: (natural language action, pyautogui action string list)
        """
        # Apply smart resize preprocessing
        processed_screenshot = self._preprocess_image_with_smart_resize(current_screenshot)
        
        prompt = self.prompt_construction(task, processed_screenshot)
        results = self.inference(prompt)

        if not results:
            print("Error: Inference returned no results in step.")
            return "Inference failed", []

        # Use resized dimensions for action revision
        image_size = results.get("image_size", {})
        width = image_size.get("width", processed_screenshot.width)
        height = image_size.get("height", processed_screenshot.height)

        natural_language_action, pyautogui_action_strings = self.parse_results(results, width, height)

        return natural_language_action, pyautogui_action_strings

    def parse_results(self, results: Dict[str, Any], width: int, height: int) -> Tuple[str, List[str]]:
        """
        Parse inference results.
        
        Args:
            results: Inference results
            width: Image width
            height: Image height
            
        Returns:
            Tuple: (natural language action, action string list)
        """
        pred_text = results.get("pred_text", "")
        natural_language_action = self.parse_natural_language_action(pred_text)
        pyautogui_actions_dicts = self.parse_pyautogui_calls(pred_text)

        actions_string_list = []
        for action_dict in pyautogui_actions_dicts:
            revised_action_dict = self.action_revision(action_dict, width, height)
            action_str = self.function_call_to_string(revised_action_dict)
            actions_string_list.append(action_str)

        return natural_language_action, actions_string_list

    def action_revision(self, action: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        """
        Revise action parameters (coordinate scaling, etc.).
        Note: With smart_resize, coordinates are already transformed to original dimensions,
        so we need to be careful about double scaling.
        
        Args:
            action: Action dictionary
            width: Image width (this is the resized width from model response)
            height: Image height (this is the resized height from model response)
            
        Returns:
            Dict: Revised action dictionary
        """
        func = action.get("function")
        kw_args = action.get("args", {})
        pos_args = action.get("*args", [])

        revised_action = {"function": func, "args": kw_args.copy(), "*args": list(pos_args)}

        mouse_actions = ["click", "doubleClick", "rightClick", "moveTo", "dragTo", "dragRel"]
        scroll_actions = ["scroll", "hscroll", "vscroll"]

        if func in mouse_actions:
            scaled_coords = False
            if 'x' in revised_action["args"] and 'y' in revised_action["args"]:
                if isinstance(revised_action["args"]['x'], (float, int)) and \
                        isinstance(revised_action["args"]['y'], (float, int)):
                    # If we're using smart_resize, coordinates are already in original dimensions
                    if self.use_smart_resize:
                        # Coordinates are already transformed, no additional scaling needed
                        revised_action["args"]["x"] = int(revised_action["args"]["x"])
                        revised_action["args"]["y"] = int(revised_action["args"]["y"])
                    else:
                        # Traditional scaling for backward compatibility
                        revised_action["args"]["x"] = int(revised_action["args"]["x"] * width)
                        revised_action["args"]["y"] = int(revised_action["args"]["y"] * height)
                    scaled_coords = True

            if not scaled_coords and len(revised_action["*args"]) >= 2:
                try:
                    x_pos = revised_action["*args"][0]
                    y_pos = revised_action["*args"][1]
                    if isinstance(x_pos, (float, int)) and isinstance(y_pos, (float, int)):
                        if self.use_smart_resize:
                            # Coordinates are already transformed, no additional scaling needed
                            revised_action["*args"][0] = int(x_pos)
                            revised_action["*args"][1] = int(y_pos)
                        else:
                            # Traditional scaling for backward compatibility
                            revised_action["*args"][0] = int(x_pos * width)
                            revised_action["*args"][1] = int(y_pos * height)
                except (IndexError, TypeError):
                    pass

        if func in scroll_actions:
            page_val = None
            if "page" in revised_action["args"]:
                page_val = revised_action["args"].pop("page", 0)

            if page_val is not None:
                clicks = int(page_val * 1000)
                revised_action["*args"] = [clicks] + revised_action["*args"]

        if func == "hotkey":
            keys_to_press = []
            if revised_action["*args"]:
                keys_to_press = [str(k).lower() for k in revised_action["*args"]]
            elif "keys" in revised_action["args"]:
                keys_value = revised_action["args"].pop("keys")
                if isinstance(keys_value, list):
                    keys_to_press = [str(k).lower() for k in keys_value]
                else:
                    keys_to_press = [str(keys_value).lower()]

            revised_action["*args"] = keys_to_press

        return revised_action

    @staticmethod
    def function_call_to_string(function_call: Dict, prefix="pyautogui.") -> str:
        """
        Convert function call dictionary to string.
        
        Args:
            function_call: Function call dictionary
            prefix: Function prefix
            
        Returns:
            str: Function call string
        """
        func_name = function_call.get("function", "")
        keyword_args_dict = function_call.get("args", {})
        positional_args_list = function_call.get("*args", []) or []

        def format_value(val):
            if isinstance(val, str):
                return json.dumps(val)
            return str(val)

        pos_args_str = ", ".join(format_value(arg) for arg in positional_args_list)
        kw_args_str = ", ".join(f"{k}={format_value(v)}" for k, v in keyword_args_dict.items())

        all_args_parts = []
        if pos_args_str:
            all_args_parts.append(pos_args_str)
        if kw_args_str:
            all_args_parts.append(kw_args_str)

        all_args = ", ".join(all_args_parts)
        return f"{prefix}{func_name}({all_args})"


# Compatibility function
def load_model(api_url: str = "http://20.91.255.106:19803/evaluate", model_name: str = "gui_actor", use_smart_resize: bool = True):
    """
    Load and return GUI Actor model instance.
    
    Args:
        api_url: API endpoint URL
        model_name: Model name
        use_smart_resize: Whether to use smart_resize for better coordinate accuracy
        
    Returns:
        GuiActor: Model instance
    """
    return GuiActor(api_url, model_name, use_smart_resize)


# Example usage
if __name__ == "__main__":
    model = GuiActor()
    
    # Test prediction functionality
    test_instruction = "Click the red button to dismiss the warning message."
    test_image = "test_image.png"  # Requires actual image file
    
    # Note: This requires actual image file and running server to test
    # response = model.predict("", test_instruction, test_image)
    # print(f"Response: {response}")
