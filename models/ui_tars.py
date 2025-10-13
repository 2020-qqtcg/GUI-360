import argparse
import os
import json
import base64
import logging
import re
import numpy as np
from typing import Tuple, List
from PIL import Image
from openai import OpenAI

from models.base import BaseModel


class UITars(BaseModel):
    """
    UI-TARS model class for GUI grounding and action prediction tasks using vLLM deployed model.
    Based on the official UI-TARS implementation with async OpenAI client.
    """

    def __init__(self, model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B", api_url: str = "http://20.91.255.106:19806/v1", api_key: str = "empty", use_smart_resize: bool = True):
        """
        Initialize UI-TARS model.

        Args:
            model_name: Model name (default: ByteDance-Seed/UI-TARS-1.5-7B)
            api_url: vLLM service API URL (default: http://20.91.255.106:19803/v1)
            api_key: API key for vLLM service (default: empty)
            use_smart_resize: Whether to use smart_resize for better coordinate accuracy
        """
        super().__init__(model_name)
        self.api_url = api_url
        self.api_key = api_key
        self.use_smart_resize = use_smart_resize
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store original image dimensions for coordinate transformation
        self.original_dimensions = None
        self.resized_dimensions = None

        # Initialize sync OpenAI client with timeout
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=60.0,  # 60秒超时
        )

        self.logger.info(f"Initialized UI-TARS model: {model_name}")
        self.logger.info(f"API URL: {api_url}")
        self.logger.info(f"Smart resize enabled: {use_smart_resize}")

    def smart_resize(self, height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280) -> Tuple[int, int]:
        """
        Adjust image dimensions to be multiples of factor and within pixel range.

        Args:
            height: Original image height
            width: Original image width
            factor: Factor to make dimensions multiples of (default: 28)
            min_pixels: Minimum number of pixels (default: 56*56)
            max_pixels: Maximum number of pixels (default: 14*14*4*1280)

        Returns:
            Tuple[int, int]: Adjusted (height, width)
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
            resized_height, resized_width = self.smart_resize(original_height, original_width)
            self.resized_dimensions = (resized_width, resized_height)

            # Resize image if dimensions changed
            if (resized_width, resized_height) != (original_width, original_height):
                resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
                self.logger.info(f"Smart resize applied: {original_width}x{original_height} -> {resized_width}x{resized_height}")
                return resized_image
            else:
                self.logger.info(f"No resize needed: {original_width}x{original_height}")
                return image

        except Exception as e:
            self.logger.error(f"Error in smart resize preprocessing: {e}")
            # Fall back to original image
            self.original_dimensions = image.size
            self.resized_dimensions = image.size
            return image

    def _transform_coordinates_to_original(self, coordinates: List[float]) -> List[float]:
        """
        Transform coordinates from resized image back to original image coordinates.

        Args:
            coordinates: [x, y] coordinates from model output (absolute pixel positions on resized image)

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

            self.logger.info(f"Coordinate transformation: {coordinates} -> {transformed_coords}")
            self.logger.info(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
            return transformed_coords

        except Exception as e:
            self.logger.error(f"Error in coordinate transformation: {e}")
            return coordinates

    def encode_image(self, image_path: str) -> str:
        """Encode image as a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def get_grounding_prompt_template(self) -> str:
        """Get the UI-TARS prompt template optimized for grounding tasks."""
        return r"""You are a GUI agent specialized in element localization. Given a screenshot and a description of what to find, you need to locate the precise coordinates of the target element.

## Task
You will be given:
1. A screenshot of a GUI interface
2. A description of an element to locate (thought)

Your task is to find the exact coordinates of the described element and click on it.

## Output Format
```
Thought: [Your reasoning about where the element is located]
Action: click(start_box='<|box_start|>(x,y)<|box_end|>')
```

## Guidelines
- Analyze the screenshot carefully to locate the target element
- Use English in the `Thought` part
- Explain your reasoning for the location in one sentence
- Output the coordinates in the exact format: click(start_box='<|box_start|>(x,y)<|box_end|>')
- Coordinates should be the center point of the target element
- If the element is not clearly visible, choose the most likely location

## Element to locate:
"""

    def format_grounding_messages(self, instruction: str, base64_image: str) -> List[dict]:
        """Format OpenAI request messages for UI-TARS grounding task."""
        prompt = self.get_grounding_prompt_template()
        
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {
                        "type": "text", 
                        "text": prompt + instruction
                    },
                ],
            },
        ]

    def sync_predict(self, instruction: str, image_path: str, temperature: float = 0.0) -> str:
        """
        Sync prediction method for UI-TARS.

        Args:
            instruction: Task instruction or grounding description
            image_path: Path to the screenshot image
            temperature: Temperature parameter for generation

        Returns:
            str: Model response containing action and coordinates
        """
        try:
            # Load and process image
            image = Image.open(image_path)
            image = image.convert("RGB")

            # Apply smart resize preprocessing
            processed_image = self._preprocess_image_with_smart_resize(image)

            # Encode processed image to base64
            from io import BytesIO
            buffer = BytesIO()
            processed_image.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Format request
            messages = self.format_grounding_messages(instruction, base64_image)

            # Call model API
            completion = self.client.chat.completions.create(
                model=self.model_name,  # Use the served model name
                messages=messages,
                temperature=temperature,
                frequency_penalty=1,
                max_tokens=4096,
            )

            # Get model response
            response_text = completion.choices[0].message.content.strip()

            return response_text

        except Exception as e:
            self.logger.error(f"Error during sync prediction: {e}")
            return f"Error: {str(e)}"

    def predict(self, system_prompt: str, user_prompt: str, image_path: str, temperature: float = 0.0, *args, **kwargs) -> str:
        """
        Universal predict method that takes system and user prompts.
        
        Args:
            system_prompt: System prompt (not used by UI-TARS, included in template)
            user_prompt: User prompt containing the instruction
            image_path: Path to the screenshot image
            temperature: Temperature parameter for generation
            
        Returns:
            str: Model response containing action and coordinates
        """
        try:
            # For UI-TARS, the user_prompt should contain the instruction
            instruction = user_prompt
            
            # Use synchronous prediction to avoid async issues
            result = self.sync_predict(instruction, image_path, temperature)
            return result
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return f"Error: {str(e)}"

    def construct_grounding_prompt(self, thought: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for grounding task.
        
        Args:
            thought: The description of what to find/click (this is the target element description)
            resolution: Screen resolution (width, height)
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # UI-TARS uses a combined prompt template, so we put everything in user_prompt
        system_prompt = ""
        # The thought IS the element description we want to locate
        user_prompt = thought
        return system_prompt, user_prompt

    def construct_action_prompt(self, instruction: str, history: str, actions: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for action prediction task.
        
        Args:
            instruction: Task instruction
            history: Action history
            actions: Available actions
            resolution: Screen resolution
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = ""
        user_prompt = f"{instruction}\n\nAction History:\n{history}\n\nAvailable Actions:\n{actions}"
        return system_prompt, user_prompt

    def parse_coordinates(self, response: str) -> List[float]:
        """
        Parse the coordinates from the UI-TARS response.
        UI-TARS outputs coordinates in relative format (0-1000 scale) within action syntax.
        
        Args:
            response: Model response containing action with coordinates
            
        Returns:
            List[float]: Parsed absolute coordinates [x, y]
        """
        try:
            response = response.strip()
            
            # Look for action with coordinates in format: click(start_box='<|box_start|>(x,y)<|box_end|>')
            # or similar patterns, supporting both integers and floats
            coord_patterns = [
                r'<\|box_start\|>\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)<\|box_end\|>',  # Official format with float support
                r'\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)',  # Simple format with float support
                r'start_box=.*?\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)',  # start_box format with float support
                r'click\(start_box=.*?\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)',  # click action format
            ]
            
            for pattern in coord_patterns:
                match = re.search(pattern, response)
                if match:
                    x_rel, y_rel = float(match.group(1)), float(match.group(2))
                    
                    # UI-TARS uses relative coordinates (0-1000 scale)
                    # Convert to absolute coordinates by dividing by 1000
                    # Note: This assumes the model outputs relative coordinates
                    # If the model outputs absolute coordinates, this conversion might not be needed
                    if x_rel <= 1000 and y_rel <= 1000:
                        # These are likely relative coordinates, keep them as is for now
                        # The evaluation framework will handle the conversion if needed
                        coords = [float(x_rel), float(y_rel)]
                        # Transform coordinates back to original image dimensions
                        return self._transform_coordinates_to_original(coords)
                    else:
                        # These might be absolute coordinates
                        coords = [float(x_rel), float(y_rel)]
                        # Transform coordinates back to original image dimensions
                        return self._transform_coordinates_to_original(coords)
            
            # If no coordinate pattern found, try to extract numbers (including floats) from the response
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if len(numbers) >= 2:
                x, y = float(numbers[0]), float(numbers[1])
                coords = [x, y]
                # Transform coordinates back to original image dimensions
                return self._transform_coordinates_to_original(coords)
                    
        except Exception as e:
            self.logger.error(f"Failed to parse coordinates from response '{response}': {e}")
            
        # Return default coordinates if parsing fails
        coords = [0.0, 0.0]
        # Transform coordinates back to original image dimensions
        return self._transform_coordinates_to_original(coords)

    def parse_thoughts(self, response: str) -> str:
        """
        Parse the reasoning thoughts from the UI-TARS response.
        UI-TARS outputs thoughts in the "Thought: ..." format.
        
        Args:
            response: Model response
            
        Returns:
            str: Extracted thoughts/reasoning
        """
        try:
            response = response.strip()
            
            # Look for "Thought:" pattern
            thought_match = re.search(r'Thought:\s*(.*?)(?:\n|Action:|$)', response, re.DOTALL)
            if thought_match:
                return thought_match.group(1).strip()
            
            # If no explicit thought pattern, return the beginning of the response
            lines = response.split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('Action:'):
                    return line.strip()
                    
        except Exception as e:
            self.logger.error(f"Failed to parse thoughts from response '{response}': {e}")
            
        return ""

    def parse_action(self, response: str):
        """
        Parse the action from the UI-TARS response.
        
        Args:
            response: Model response
            
        Returns:
            tuple: (function_name, args_dict, status)
        """
        try:
            response = response.strip()
            
            # Look for action pattern like: click(start_box='<|box_start|>(x,y)<|box_end|>')
            action_patterns = [
                r'Action:\s*(\w+)\((.*?)\)',  # Action: function(args)
                r'(\w+)\((.*?)\)',  # function(args)
            ]
            
            for pattern in action_patterns:
                match = re.search(pattern, response)
                if match:
                    function_name = match.group(1)
                    args_str = match.group(2)
                    
                    # Parse arguments
                    args_dict = {}
                    if 'start_box' in args_str:
                        coord_match = re.search(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', args_str)
                        if coord_match:
                            x, y = int(coord_match.group(1)), int(coord_match.group(2))
                            coords = [float(x), float(y)]
                            # Transform coordinates back to original image dimensions
                            args_dict['start_coordinate'] = self._transform_coordinates_to_original(coords)

                    if 'end_box' in args_str:
                        end_coord_match = re.search(r'end_box=.*?<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', args_str)
                        if end_coord_match:
                            x, y = int(end_coord_match.group(1)), int(end_coord_match.group(2))
                            coords = [float(x), float(y)]
                            # Transform coordinates back to original image dimensions
                            args_dict['end_coordinate'] = self._transform_coordinates_to_original(coords)
                    
                    if 'content' in args_str:
                        content_match = re.search(r"content=['\"]([^'\"]*)['\"]", args_str)
                        if content_match:
                            args_dict['content'] = content_match.group(1)
                    
                    if 'key' in args_str:
                        key_match = re.search(r"key=['\"]([^'\"]*)['\"]", args_str)
                        if key_match:
                            args_dict['key'] = key_match.group(1)
                    
                    return function_name, args_dict, "success"
                    
        except Exception as e:
            self.logger.error(f"Failed to parse action from response '{response}': {e}")
            
        return None, None, None


if __name__ == "__main__":
    # Example usage
    ui_tars = UITars(use_smart_resize=True)  # Enable smart resize for better coordinate accuracy

    # Test prediction
    system_prompt = ""
    user_prompt = "Find and click on the login button"
    image_path = "test_image.png"

    try:
        result = ui_tars.predict(system_prompt, user_prompt, image_path)
        print(f"UI-TARS prediction: {result}")

        # Parse components
        coords = ui_tars.parse_coordinates(result)
        thoughts = ui_tars.parse_thoughts(result)
        action_info = ui_tars.parse_action(result)

        print(f"Parsed coordinates: {coords}")
        print(f"Parsed thoughts: {thoughts}")
        print(f"Parsed action: {action_info}")

    except Exception as e:
        print(f"Error during test: {e}")
