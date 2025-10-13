import argparse
import os
import json
import base64
import logging
from typing import Tuple, List
from PIL import Image
from openai import OpenAI

from models.base import BaseModel


class UGround(BaseModel):
    """
    UGround model class for GUI grounding tasks using vLLM deployed model.
    Based on the official UGround implementation with async OpenAI client.
    """

    def __init__(self, model_name: str = "osunlp/UGround-V1-7B", api_url: str = "http://20.91.255.106:19803/v1", api_key: str = "token-abc123"):
        """
        Initialize UGround model.
        
        Args:
            model_name: Model name (default: osunlp/UGround-V1-7B)
            api_url: vLLM service API URL (default: http://20.91.255.106:19803/v1)
            api_key: API key for vLLM service (default: token-abc123)
        """
        super().__init__(model_name)
        self.api_url = api_url
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize sync OpenAI client with timeout
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=60.0,  # 60秒超时
        )
        
        self.logger.info(f"Initialized UGround model: {model_name}")
        self.logger.info(f"API URL: {api_url}")

    def encode_image(self, image_path: str) -> str:
        """Encode image as a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def format_openai_template(self, description: str, base64_image: str) -> List[dict]:
        """Format OpenAI request template for UGround."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {description}

Answer:""",
                    },
                ],
            },
        ]

    def sync_predict(self, description: str, image_path: str, temperature: float = 0.0) -> str:
        """
        Sync prediction method for UGround.
        
        Args:
            description: Description of the element to locate
            image_path: Path to the screenshot image
            temperature: Temperature parameter for generation
            
        Returns:
            str: Model response containing coordinates
        """
        try:
            # Load and process image
            image = Image.open(image_path)
            image = image.convert("RGB")
            width, height = image.size

            # Encode image to base64
            base64_image = self.encode_image(image_path)

            # Format request
            messages = self.format_openai_template(description, base64_image)

            # Call model API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )

            # Get model response
            response_text = completion.choices[0].message.content.strip()
            
            # Parse coordinates from response
            try:
                # UGround returns coordinates in ratio format (0-1000)
                ratio_coords = eval(response_text)
                x_ratio, y_ratio = ratio_coords

                # Convert to absolute coordinates
                x_coord = int(x_ratio / 1000 * width)
                y_coord = int(y_ratio / 1000 * height)

                return f"({x_coord}, {y_coord})"
                
            except Exception as parse_error:
                self.logger.error(f"Error parsing coordinates from response '{response_text}': {parse_error}")
                # Return raw response if parsing fails
                return response_text

        except Exception as e:
            self.logger.error(f"Error during sync prediction: {e}")
            return f"Error: {str(e)}"

    def predict(self, system_prompt: str, user_prompt: str, image_path: str, temperature: float = 0.0, *args, **kwargs) -> str:
        """
        Universal predict method that takes system and user prompts.
        
        Args:
            system_prompt: System prompt (not used by UGround)
            user_prompt: User prompt containing the description
            image_path: Path to the screenshot image
            temperature: Temperature parameter for generation
            
        Returns:
            str: Model response containing coordinates
        """
        try:
            # For UGround, the user_prompt should contain the description
            description = user_prompt
            
            # Use synchronous prediction to avoid async issues
            result = self.sync_predict(description, image_path, temperature)
            return result
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return f"Error: {str(e)}"

    def construct_grounding_prompt(self, thought: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for grounding task.
        
        Args:
            thought: The description of what to find
            resolution: Screen resolution (not used by UGround)
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # UGround doesn't use separate system prompts, everything is in user prompt
        system_prompt = ""
        user_prompt = thought
        return system_prompt, user_prompt

    def construct_action_prompt(self, instruction: str, history: str, actions: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """
        Construct system and user prompts for action prediction task.
        UGround is primarily designed for grounding, not action prediction.
        
        Args:
            instruction: Task instruction
            history: Action history
            actions: Available actions
            resolution: Screen resolution
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # UGround is not designed for action prediction, but we provide a basic implementation
        system_prompt = ""
        user_prompt = f"Instruction: {instruction}\nHistory: {history}\nActions: {actions}"
        return system_prompt, user_prompt

    def parse_coordinates(self, response: str) -> List[float]:
        """
        Parse the coordinates from the UGround response.
        
        Args:
            response: Model response containing coordinates
            
        Returns:
            List[float]: Parsed coordinates [x, y]
        """
        try:
            response = response.strip()
            
            # UGround returns coordinates in format "(x, y)"
            if response.startswith("(") and response.endswith(")"):
                # Remove parentheses and split by comma
                coords_str = response[1:-1]
                x_str, y_str = coords_str.split(",")
                x = float(x_str.strip())
                y = float(y_str.strip())
                return [x, y]
            else:
                # Try to evaluate the response directly (for ratio format)
                coords = eval(response)
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    return [float(coords[0]), float(coords[1])]
                    
        except Exception as e:
            self.logger.error(f"Failed to parse coordinates from response '{response}': {e}")
            
        # Return default coordinates if parsing fails
        return [0.0, 0.0]

    def parse_thoughts(self, response: str) -> str:
        """
        Parse the reasoning thoughts from the model response.
        UGround doesn't provide reasoning thoughts, so return empty string.
        
        Args:
            response: Model response
            
        Returns:
            str: Empty string (UGround doesn't provide thoughts)
        """
        return ""

    def parse_action(self, response: str):
        """
        Parse the action from the model response.
        UGround is not designed for action prediction.
        
        Args:
            response: Model response
            
        Returns:
            tuple: (None, None, None) as UGround doesn't predict actions
        """
        return None, None, None


if __name__ == "__main__":
    # Example usage
    uground = UGround()
    
    # Test prediction
    system_prompt = ""
    user_prompt = "Find the login button"
    image_path = "test_image.png"
    
    try:
        result = uground.predict(system_prompt, user_prompt, image_path)
        print(f"UGround prediction: {result}")
        
        # Parse coordinates
        coords = uground.parse_coordinates(result)
        print(f"Parsed coordinates: {coords}")
        
    except Exception as e:
        print(f"Error during test: {e}")
