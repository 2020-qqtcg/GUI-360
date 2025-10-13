import base64
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional

import requests
from PIL import Image

from models.base import BaseModel


class Aguvis7B720P(BaseModel):
    """
    Aguvis-7B-720P model class for GUI agent tasks.
    Supports both grounding and action prediction through different inference modes.
    """

    def __init__(self, api_url: str = "http://20.91.255.106:19806/evaluate", model_name: str = "xlangai/Aguvis-7B-720P", use_smart_resize: bool = True):
        """
        Initialize Aguvis-7B-720P model.
        
        Args:
            api_url: API endpoint for Aguvis service
            model_name: Model name
            use_smart_resize: Whether to use smart resize (placeholder for future use)
        """
        super().__init__(model_name)
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}
        self.use_smart_resize = use_smart_resize
        self._current_image_resolution = None  # Store current image resolution for coordinate conversion
        
        print(f"Initialized Aguvis-7B-720P model")
        print(f"API URL: {self.api_url}")
        print(f"Model: {self.model_name}")

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

    def _encode_pil_image(self, image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            byte_data = buffer.getvalue()
            encoded_string = base64.b64encode(byte_data).decode("utf-8")
            return encoded_string
        except Exception as e:
            print(f"Error encoding PIL image: {e}")
            raise

    def predict(self, system_prompt: str, user_prompt: str, image_path: str, max_tokens: int = 1024, temperature: float = 0.0, *args, **kwargs) -> str:
        """
        Universal prediction method that accepts system and user prompts.
        
        Args:
            system_prompt: System prompt (not directly used by Aguvis API)
            user_prompt: User prompt containing task instructions
            image_path: Path to image file
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            
        Returns:
            str: Model response text
        """
        try:
            # Load and encode image
            if isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
                # If it's a URL, download the image
                response = requests.get(image_path)
                screenshot = Image.open(BytesIO(response.content))
                image_b64 = self._encode_pil_image(screenshot)
                # Store resolution for coordinate conversion
                self._current_image_resolution = screenshot.size
            else:
                # Load image to get resolution
                screenshot = Image.open(image_path)
                self._current_image_resolution = screenshot.size
                image_b64 = self._encode_image(image_path)
            
            # Build request data for general prediction (using self-plan mode)
            request_data = {
                "image": image_b64,
                "user_instruction": user_prompt,
                "previous_actions": None,
                "low_level_instruction": None,
                "mode": "self-plan",
                "temperature": temperature,
                "max_new_tokens": max_tokens
            }
            
            # Make API call
            response_text = self._make_api_call(request_data)
            return response_text
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"Error: {str(e)}"

    def construct_grounding_prompt(self, thought: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for grounding task.
        
        Args:
            thought: Thought content (element description to locate)
            resolution: Screen resolution
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # For Aguvis, we'll use the grounding mode which expects the thought as the instruction
        system_prompt = "You are a GUI grounding agent. Locate the specified UI element in the screenshot."
        user_prompt = f"Please locate the element: {thought}"
        return system_prompt, user_prompt

    def construct_action_prompt(self, instruction: str, history: str, actions: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for action prediction task.
        
        Args:
            instruction: Task instruction
            history: History of previous actions
            actions: Supported actions (not directly used by Aguvis)
            resolution: Screen resolution
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = "You are a GUI action prediction agent. Determine the next action to take."
        user_prompt = f"Task: {instruction}\nPrevious actions: {history}"
        return system_prompt, user_prompt

    def predict_grounding(self, instruction: str, image_path: str, resolution: tuple[int, int] = (1920, 1080)) -> str:
        """
        Predict grounding coordinates for a given instruction.
        
        Args:
            instruction: Instruction describing the element to locate
            image_path: Path to screenshot image
            resolution: Screen resolution
            
        Returns:
            str: Model response containing grounding information
        """
        try:
            # Load and encode image
            if isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
                response = requests.get(image_path)
                screenshot = Image.open(BytesIO(response.content))
                image_b64 = self._encode_pil_image(screenshot)
                # Store resolution for coordinate conversion
                self._current_image_resolution = screenshot.size
            else:
                # Load image to get resolution
                screenshot = Image.open(image_path)
                self._current_image_resolution = screenshot.size
                image_b64 = self._encode_image(image_path)
            
            # Build request data for grounding task
            request_data = {
                "image": image_b64,
                "user_instruction": instruction,
                "previous_actions": None,
                "low_level_instruction": None,
                "mode": "grounding",  # Use grounding mode specifically
                "temperature": 0.0,
                "max_new_tokens": 1024
            }
            
            # Make API call
            response_text = self._make_api_call(request_data)
            return response_text
            
        except Exception as e:
            print(f"Error during grounding prediction: {e}")
            return f"Error: {str(e)}"

    def predict_action(self, instruction: str, history: List[str], image_path: str, resolution: tuple[int, int] = (1920, 1080)) -> str:
        """
        Predict next action for a given instruction and history.
        
        Args:
            instruction: Task instruction
            history: List of previous actions
            image_path: Path to screenshot image
            resolution: Screen resolution
            
        Returns:
            str: Model response containing action information
        """
        try:
            # Load and encode image
            if isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
                response = requests.get(image_path)
                screenshot = Image.open(BytesIO(response.content))
                image_b64 = self._encode_pil_image(screenshot)
                # Store resolution for coordinate conversion
                self._current_image_resolution = screenshot.size
            else:
                # Load image to get resolution
                screenshot = Image.open(image_path)
                self._current_image_resolution = screenshot.size
                image_b64 = self._encode_image(image_path)
            
            # Build request data for action prediction
            request_data = {
                "image": image_b64,
                "user_instruction": instruction,
                "previous_actions": history if history else None,
                "low_level_instruction": None,
                "mode": "self-plan",  # Use self-plan mode for action prediction
                "temperature": 0.0,
                "max_new_tokens": 1024
            }
            
            # Make API call
            response_text = self._make_api_call(request_data)
            return response_text
            
        except Exception as e:
            print(f"Error during action prediction: {e}")
            return f"Error: {str(e)}"

    def _make_api_call(self, request_data: Dict[str, Any]) -> str:
        """
        Make API call to Aguvis server.
        
        Args:
            request_data: Request data dictionary
            
        Returns:
            str: Response text from the server
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    data=json.dumps(request_data), 
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract response_text from the API response
                if isinstance(result, dict) and "response_text" in result:
                    return result["response_text"]
                else:
                    print(f"Unexpected response format: {result}")
                    return str(result)
                    
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code in [500, 502, 503, 504]:
                    retry_count += 1
                    print(f"Retrying... ({retry_count}/{max_retries})")
                else:
                    raise
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                retry_count += 1
                print(f"Retrying... ({retry_count}/{max_retries})")
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON response: {e}")
                if 'response' in locals():
                    print(f"Response text: {response.text}")
                raise
        
        raise RuntimeError("Max retries reached. Unable to get a valid response.")

    def parse_coordinates(self, response: str, resolution: tuple[int, int] = None) -> list[float]:
        """
        Parse coordinates from Aguvis model response and convert to absolute coordinates.
        Aguvis typically outputs relative coordinates (0-1 range) that need to be converted to absolute coordinates.
        
        Args:
            response: Model response text
            resolution: Screen resolution (width, height) for coordinate conversion
            
        Returns:
            list[float]: Parsed absolute coordinates [x, y]
        """
        try:
            # Use provided resolution, stored resolution, or default
            if resolution is None:
                if self._current_image_resolution is not None:
                    resolution = self._current_image_resolution
                else:
                    resolution = (1920, 1080)
            
            width, height = resolution
            
            # First try to find coordinates in <click> tags (common Aguvis format)
            click_pattern = r'<click>(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)</click>'
            click_match = re.search(click_pattern, response)
            if click_match:
                x, y = float(click_match.group(1)), float(click_match.group(2))
                # Check if coordinates are relative (0-1 range) and convert to absolute
                if 0 <= x <= 1 and 0 <= y <= 1:
                    x = x * width
                    y = y * height
                return [x, y]
            
            # Try to find coordinates in brackets [x, y]
            bracket_pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
            bracket_match = re.search(bracket_pattern, response)
            if bracket_match:
                x, y = float(bracket_match.group(1)), float(bracket_match.group(2))
                # Check if coordinates are relative (0-1 range) and convert to absolute
                if 0 <= x <= 1 and 0 <= y <= 1:
                    x = x * width
                    y = y * height
                return [x, y]
            
            # Try to find coordinates in parentheses (x, y)
            paren_pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
            paren_match = re.search(paren_pattern, response)
            if paren_match:
                x, y = float(paren_match.group(1)), float(paren_match.group(2))
                # Check if coordinates are relative (0-1 range) and convert to absolute
                if 0 <= x <= 1 and 0 <= y <= 1:
                    x = x * width
                    y = y * height
                return [x, y]
            
            # Try to parse as JSON if the response looks like JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                try:
                    data = json.loads(response)
                    if "coordinates" in data:
                        coords = data["coordinates"]
                        if isinstance(coords, list) and len(coords) >= 2:
                            x, y = float(coords[0]), float(coords[1])
                            # Check if coordinates are relative (0-1 range) and convert to absolute
                            if 0 <= x <= 1 and 0 <= y <= 1:
                                x = x * width
                                y = y * height
                            return [x, y]
                except json.JSONDecodeError:
                    pass
            
            # If no pattern matches, try to extract any two numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if len(numbers) >= 2:
                x, y = float(numbers[0]), float(numbers[1])
                # Check if coordinates are relative (0-1 range) and convert to absolute
                if 0 <= x <= 1 and 0 <= y <= 1:
                    x = x * width
                    y = y * height
                return [x, y]
            
            # Default fallback
            print(f"Warning: Could not parse coordinates from response: {response}")
            return [0.0, 0.0]
            
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
            return [0.0, 0.0]

    def parse_thoughts(self, response: str) -> str:
        """
        Parse reasoning thoughts from Aguvis model response.
        
        Args:
            response: Model response text
            
        Returns:
            str: Parsed thoughts/reasoning
        """
        try:
            # Try to extract thoughts from JSON format
            if response.strip().startswith('{') and response.strip().endswith('}'):
                try:
                    data = json.loads(response)
                    if "thoughts" in data:
                        return data["thoughts"]
                    elif "reasoning" in data:
                        return data["reasoning"]
                except json.JSONDecodeError:
                    pass
            
            # Try to find thoughts in specific tags
            thought_patterns = [
                r'<thought>(.*?)</thought>',
                r'Thought:\s*(.*?)(?:\n|$)',
                r'Reasoning:\s*(.*?)(?:\n|$)',
                r'Analysis:\s*(.*?)(?:\n|$)'
            ]
            
            for pattern in thought_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # If no specific format found, return first few lines as thoughts
            lines = response.strip().split('\n')
            if lines:
                return lines[0]
            
            return response
            
        except Exception as e:
            print(f"Error parsing thoughts: {e}")
            return ""

    def parse_action(self, response: str, resolution: tuple[int, int] = None) -> Tuple[str, Dict[str, Any], str]:
        """
        Parse action information from Aguvis model response.
        
        Args:
            response: Model response text
            resolution: Screen resolution (width, height) for coordinate conversion
            
        Returns:
            Tuple: (function_name, args_dict, status)
        """
        try:
            # Use provided resolution, stored resolution, or default
            if resolution is None:
                if self._current_image_resolution is not None:
                    resolution = self._current_image_resolution
                else:
                    resolution = (1920, 1080)
            
            width, height = resolution
            
            # Try to parse action from common Aguvis action formats
            
            # Pattern 1: <click>x,y</click>
            click_pattern = r'<click>(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)</click>'
            click_match = re.search(click_pattern, response)
            if click_match:
                x, y = float(click_match.group(1)), float(click_match.group(2))
                # Convert relative coordinates to absolute if needed
                if 0 <= x <= 1 and 0 <= y <= 1:
                    x = x * width
                    y = y * height
                return "click", {"coordinate": [int(x), int(y)]}, "CONTINUE"
            
            # Pattern 2: Action: click(x, y)
            action_pattern = r'Action:\s*(\w+)\(([^)]+)\)'
            action_match = re.search(action_pattern, response, re.IGNORECASE)
            if action_match:
                function_name = action_match.group(1).lower()
                args_str = action_match.group(2)
                
                # Parse arguments
                args_dict = {}
                if function_name == "click":
                    # Extract coordinates (handle decimal numbers)
                    coords = re.findall(r'\d+(?:\.\d+)?', args_str)
                    if len(coords) >= 2:
                        x, y = float(coords[0]), float(coords[1])
                        # Convert relative coordinates to absolute if needed
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            x = x * width
                            y = y * height
                        args_dict["coordinate"] = [int(x), int(y)]
                elif function_name == "type":
                    # Extract coordinates and text
                    coords = re.findall(r'\d+(?:\.\d+)?', args_str)
                    if len(coords) >= 2:
                        x, y = float(coords[0]), float(coords[1])
                        # Convert relative coordinates to absolute if needed
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            x = x * width
                            y = y * height
                        args_dict["coordinate"] = [int(x), int(y)]
                    # Extract text (assume it's in quotes)
                    text_match = re.search(r'["\']([^"\']*)["\']', args_str)
                    if text_match:
                        args_dict["keys"] = text_match.group(1)
                
                return function_name, args_dict, "CONTINUE"
            
            # Pattern 3: JSON format
            if response.strip().startswith('{') and response.strip().endswith('}'):
                try:
                    data = json.loads(response)
                    if "tool_call" in data:
                        tool_call = data["tool_call"]
                        function_name = tool_call.get("function", "")
                        args_dict = tool_call.get("args", {}).copy()
                        status = tool_call.get("status", "CONTINUE")
                        
                        # Convert relative coordinates to absolute if present
                        if "coordinate" in args_dict and isinstance(args_dict["coordinate"], list) and len(args_dict["coordinate"]) >= 2:
                            x, y = float(args_dict["coordinate"][0]), float(args_dict["coordinate"][1])
                            if 0 <= x <= 1 and 0 <= y <= 1:
                                x = x * width
                                y = y * height
                            args_dict["coordinate"] = [int(x), int(y)]
                        
                        return function_name, args_dict, status
                except json.JSONDecodeError:
                    pass
            
            # Pattern 4: Look for specific action indicators
            if "click" in response.lower():
                coords = re.findall(r'\d+(?:\.\d+)?', response)
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    # Convert relative coordinates to absolute if needed
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        x = x * width
                        y = y * height
                    return "click", {"coordinate": [int(x), int(y)]}, "CONTINUE"
            
            if "type" in response.lower():
                coords = re.findall(r'\d+(?:\.\d+)?', response)
                text_match = re.search(r'["\']([^"\']*)["\']', response)
                args_dict = {}
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    # Convert relative coordinates to absolute if needed
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        x = x * width
                        y = y * height
                    args_dict["coordinate"] = [int(x), int(y)]
                if text_match:
                    args_dict["keys"] = text_match.group(1)
                return "type", args_dict, "CONTINUE"
            
            # Check for completion indicators
            completion_indicators = ["task completed", "finished", "done", "complete"]
            if any(indicator in response.lower() for indicator in completion_indicators):
                return "", {}, "FINISH"
            
            return None, None, None
            
        except Exception as e:
            print(f"Error parsing action from response: {e}")
            return None, None, None


# Factory function for compatibility
def load_aguvis_model(api_url: str = "http://20.91.255.106:19806/evaluate", model_name: str = "xlangai/Aguvis-7B-720P"):
    """
    Load and return Aguvis-7B-720P model instance.
    
    Args:
        api_url: API endpoint URL
        model_name: Model name
        
    Returns:
        Aguvis7B720P: Model instance
    """
    return Aguvis7B720P(api_url=api_url, model_name=model_name)


# Example usage
if __name__ == "__main__":
    # Test the model
    model = Aguvis7B720P()
    
    # Example grounding task
    test_instruction = "Click the Save button"
    test_image = "test_screenshot.png"  # Would need actual image file
    
    print("Testing Aguvis-7B-720P model...")
    print(f"Model: {model.model_name}")
    print(f"API URL: {model.api_url}")
    
    # Note: Actual testing would require a running server and valid image file
    # response = model.predict_grounding(test_instruction, test_image)
    # print(f"Grounding response: {response}")
