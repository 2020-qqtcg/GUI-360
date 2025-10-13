from abc import ABC, abstractmethod
from models.base import BaseModel
import json
from tqdm import tqdm

from .gpt import GPT_predictor
from .omniparser import OmniparserScreenParsing

class MockModel(BaseModel):

    def __init__(self, model_name: str, result_path: str):
        self.model_name = model_name
        self.results = self.load_result(result_path)

    def load_result(self, result_path: str):
        print(f"Start Loding")
        with open(result_path, "r", encoding='utf-8') as f:
            content = json.load(f)
            detail_results = content["detailed_results"]

        samples = {}
        for sample in tqdm(detail_results):
            sample_id = sample["sample_id"]
            raw_model_output = sample["raw_model_output"]
            samples[sample_id] = raw_model_output
        print(f"Load {len(samples)} samples")
        return samples


    def predict(
        self, system_prompt: str, user_prompt: str, image_path: str, *args, **kwargs
    ):
        """Universal predict method that takes system and user prompts."""
        sample_id = kwargs["sample_id"]
        raw_model_output = self.results[sample_id]
        return raw_model_output

    def construct_grounding_prompt(
        self, thought: str, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for grounding task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        return "", ""

    def construct_action_prompt(
        self, instruction: str, history: str, actions: str, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for action prediction task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        return "", ""

    def construct_action_a11y_prompt(
        self, instruction: str, history: list, actions: str, control_infos: dict
    ) -> tuple[str, str]:
        """Construct system and user prompts for A11Y action prediction task.

        Args:
            instruction: User instruction for the task
            history: List of previous actions/thoughts
            actions: Supported actions string
            control_infos: A11Y control information dictionary

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # Default implementation - subclasses can override for specific behavior
        # For models that don't support A11Y, fall back to regular action prompt
        history_str = "\n".join(history) if history else "None"
        return self.construct_action_prompt(
            instruction, history_str, actions, (1920, 1080)
        )

    def construct_screen_parsing_prompt(self, resolution) -> tuple[str, str]:
        """Construct system and user prompts for screen parsing task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # Default implementation using general screen parsing prompts
        return "", ""

    def parse_screen_parsing(self, response: str) -> list[dict]:
        """Parse screen parsing response to extract control information.

        Args:
            response: Model response containing control information

        Returns:
            list[dict]: List of control information dictionaries
        """
        # Default implementation - try to parse JSON response
        actual_model = None
        if "gpt" in self.model_name:
            actual_model = GPT_predictor("mock")

        elif "omniparser" in self.model_name:
            actual_model = OmniparserScreenParsing("mock", None)

        if actual_model:
            return actual_model.parse_screen_parsing(response)
        return []

    def parse_thoughts(self, response: str) -> str:
        """Parse thoughts/reasoning from model response.

        Args:
            response: Model response text

        Returns:
            str: Parsed thoughts or empty string if not found
        """
        # Default implementation - try to extract thoughts from various formats
        try:
            actual_model = None
            if "gpt" in self.model_name:
                actual_model = GPT_predictor("mock")

            elif "omniparser" in self.model_name:
                actual_model = OmniparserScreenParsing("mock", None)

            if actual_model:
                return actual_model.parse_thoughts(response)
            else:
                return response
        except Exception:
            return ""

    def parse_coordinates(self, response: str) -> list[float]:
        """
        Parse the coordinates from the model response.
        This is a placeholder implementation and should be replaced with actual parsing logic.
        """
        response = response.strip()
        if response.startswith("[") and response.endswith("]"):
            try:
                coords = eval(response)
                if isinstance(coords, list) and len(coords) == 2:
                    return coords
            except Exception as e:
                raise ValueError(f"Failed to parse coordinates: {e}")
        raise ValueError("Invalid format for coordinates.")

    def parse_thoughts(self, response: str) -> str:
        """
        Parse the reasoning thoughts from the model response.
        This is a placeholder implementation and should be replaced with actual parsing logic.
        """
        response = response.strip()
        if response.startswith("{") and response.endswith("}"):
            try:
                data = eval(response)
                return data.get("thoughts", "")
            except Exception as e:
                raise ValueError(f"Failed to parse thoughts: {e}")
        raise ValueError("Invalid format for thoughts.")

    def parse_action(self, response: str):
        """
        Parse the action from the model response.
        Returns a tuple of (function_name, args_dict, status) or (None, None, None) if parsing fails.
        For drag operations, args_dict will contain start_coordinate and end_coordinate.
        """
        # This is a placeholder implementation - subclasses should override this
        return None, None, None

    def parse_args(self, response: str) -> dict:
        """
        Parse the arguments from the model response.
        This is a placeholder implementation and should be replaced with actual parsing logic.
        """
        response = response.strip()
        if response.startswith("{") and response.endswith("}"):
            try:
                data = eval(response)
                return data.get("args", {})
            except Exception as e:
                raise ValueError(f"Failed to parse args: {e}")
        raise ValueError("Invalid format for args.")

    def parse_status(self, response: str) -> str:
        """
        Parse the status from the model response.
        This is a placeholder implementation and should be replaced with actual parsing logic.
        """
        response = response.strip()
        if response.startswith("{") and response.endswith("}"):
            try:
                data = eval(response)
                return data.get("status", "")
            except Exception as e:
                raise ValueError(f"Failed to parse status: {e}")
        raise ValueError("Invalid format for status.")

    def parse_control_list(self, response: str) -> list:
        """
        Parse the control list from the model response.
        This is a placeholder implementation and should be replaced with actual parsing logic.
        """
        response = response.strip()
        if response.startswith("[") and response.endswith("]"):
            try:
                controls = eval(response)
                if isinstance(controls, list):
                    return controls
            except Exception as e:
                raise ValueError(f"Failed to parse control list: {e}")
        raise ValueError("Invalid format for control list.")
