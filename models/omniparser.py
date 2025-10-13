# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import ast
import base64
import requests
from typing import Any, Dict, List, Tuple
from PIL import Image

from models.base import BaseModel


class OmniparserService:
    """Service class for calling Omniparser API server."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Omniparser service.

        Args:
            config: Configuration dictionary containing:
                - som_model_path: Path to SOM model
                - caption_model_name: Name of caption model
                - caption_model_path: Path to caption model
                - device: Device to run on
                - BOX_TRESHOLD: Box threshold for detection
                - host: Server host (default: '0.0.0.0')
                - port: Server port (default: 7861)
        """
        try:
            self.som_model_path = config.get('som_model_path', '../../weights/icon_detect/model.pt')
            self.caption_model_name = config.get('caption_model_name', 'florence2')
            self.caption_model_path = config.get('caption_model_path', '../../weights/icon_caption_florence')
            self.device = config.get('device', 'cpu')
            self.box_threshold = config.get('BOX_TRESHOLD', 0.05)
            self.host = config.get('host', '0.0.0.0')
            self.port = config.get('port', 7861)
            self.base_url = f"http://{self.host}:{self.port}"
        except Exception as e:
            pass

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('ascii')
        return encoded_string

    def chat_completion(self, image_path: str, api_name: str = "/process") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Call the Omniparser API for grounding/screen parsing.

        Args:
            image_path: Path to the image file
            box_threshold: Box threshold for detection
            iou_threshold: IoU threshold
            use_paddleocr: Whether to use PaddleOCR
            imgsz: Image size
            api_name: API endpoint name

        Returns:
            Tuple of (base64_image, parsed_content_list)
        """
        try:
            base64_image = self.encode_image_to_base64(image_path)

            request_data = {
                "base64_image": base64_image
            }

            url = f"{self.base_url}{api_name}"
            response = requests.post(url, json=request_data, timeout=120)

            if response.status_code == 200:
                result = response.json()
                som_image_base64 = result.get("som_image_base64", "")
                parsed_content_list = result.get("parsed_content_list", [])
                latency = result.get("latency", 0.0)

                print(f"Omniparser API call successful. Latency: {latency:.3f}s")

                return som_image_base64, parsed_content_list
            else:
                print(f"Omniparser API call failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return "", []

        except Exception as e:
            print(f"Error calling Omniparser API: {e}")
            return "", []


class OmniparserScreenParsing(BaseModel):
    """
    The OmniparserScreenParsing class is used for screen parsing tasks.
    It extracts all interactive control elements from screenshots.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the OmniparserScreenParsing model.

        Args:
            model_name: Name of the model
            config: Configuration dictionary for Omniparser service
        """
        super().__init__(model_name)
        self.service = OmniparserService(config)

    def predict(
        self,
        image_path: str,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Predict screen parsing for the given image.
        Returns control elements in the format expected by the evaluation framework.

        Args:
            image_path: Path to the image file
            box_threshold: Box threshold for detection
            iou_threshold: IoU threshold
            use_paddleocr: Whether to use PaddleOCR
            imgsz: Image size

        Returns:
            List of control information dictionaries with format:
            [
                {
                    "control_text": "Button text or description",
                    "control_rect": [left, top, right, bottom]
                },
                ...
            ]
        """
        if not os.path.exists(image_path):
            return []

        resolution = Image.open(image_path).size
        try:
            som_image_base64, parsed_content_list = self.service.chat_completion(
                image_path, "/parse/"
            )

            if not parsed_content_list:
                print("Warning: No control elements found by Omniparser")
                return []

            control_elements = []
            for item in parsed_content_list:
                try:
                    if isinstance(item, dict) and "bbox" in item:
                        bbox = item["bbox"]
                        content = item.get("content", "")

                        if isinstance(bbox, list) and len(bbox) == 4:
                            control_rect = bbox
                        elif isinstance(bbox, dict):
                            control_rect = [
                                bbox.get("left", 0),
                                bbox.get("top", 0),
                                bbox.get("right", 0),
                                bbox.get("bottom", 0)
                            ]
                        else:
                            continue

                        control_rect = [
                            control_rect[0] * resolution[0],
                            control_rect[1] * resolution[1],
                            control_rect[2] * resolution[0],
                            control_rect[3] * resolution[1]
                        ]

                        control_elements.append({
                            "control_text": content,
                            "control_rect": control_rect
                        })

                except Exception as e:
                    print(f"Error processing control element: {e}")
                    continue

            return control_elements

        except Exception as e:
            print(f"Warning: Failed to get screen parsing results for Omniparser. Error: {e}")
            return []

    def construct_screen_parsing_prompt(self, resolution: tuple[int, int]) -> tuple[str, str]:
        """Construct system and user prompts for screen parsing task."""
        from prompts.prompt_screen_parsing import (
            SCREEN_PARSING_SYS_PROMPT,
            SCREEN_PARSING_USER_PROMPT,
        )

        return SCREEN_PARSING_SYS_PROMPT, SCREEN_PARSING_USER_PROMPT.format(
            resolution=resolution
        )

    def construct_action_prompt(self, instruction: str, history: str, actions: str, resolution: tuple[int, int]) -> tuple[str, str]:
        return ""
    
    def construct_grounding_prompt(self, thought: str, resolution: tuple[int, int]) -> tuple[str, str]:
        return ""

    def parse_screen_parsing(self, response: Any) -> Any:
        """
        Parse screen parsing response.
        For Omniparser, the actual parsing is done in the predict method.
        This method is here for interface compatibility.

        Args:
            response: Raw response from the model (not used for Omniparser)

        Returns:
            List of control information dictionaries
        """
        return response
