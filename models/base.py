from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(
        self, system_prompt: str, user_prompt: str, image_path: str, *args, **kwargs
    ):
        """Universal predict method that takes system and user prompts."""
        pass

    @abstractmethod
    def construct_grounding_prompt(
        self, thought: str, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for grounding task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        pass

    @abstractmethod
    def construct_action_prompt(
        self, instruction: str, history: str, actions: str, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for action prediction task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        pass

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
        history_str = "\n".join(history) if history else "None"
        return self.construct_action_prompt(
            instruction, history_str, actions, (1920, 1080)
        )

    def construct_screen_parsing_prompt(self, resolution) -> tuple[str, str]:
        """Construct system and user prompts for screen parsing task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        from prompts.prompt_screen_parsing import (
            SCREEN_PARSING_SYS_PROMPT,
            SCREEN_PARSING_USER_PROMPT,
        )

        return SCREEN_PARSING_SYS_PROMPT, SCREEN_PARSING_USER_PROMPT.format(
            resolution=resolution
        )

    def parse_screen_parsing(self, response: str) -> list[dict]:
        """Parse screen parsing response to extract control information.

        Args:
            response: Model response containing control information

        Returns:
            list[dict]: List of control information dictionaries
        """
        try:
            import json
            import re

            response = response.strip()

            json_pattern = r"\[.*?\]"
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        if isinstance(parsed[0], dict) and any(
                            key in parsed[0] for key in ["control_text", "control_rect"]
                        ):
                            return parsed
                except json.JSONDecodeError:
                    continue

            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and any(
                    key in parsed for key in ["control_text", "control_rect"]
                ):
                    return [parsed]
            except json.JSONDecodeError:
                pass

            return []

        except Exception as e:
            return []

    def parse_thoughts(self, response: str) -> str:
        """Parse thoughts/reasoning from model response.

        Args:
            response: Model response text

        Returns:
            str: Parsed thoughts or empty string if not found
        """
        try:
            import json
            import re

            if response.strip().startswith("{") and response.strip().endswith("}"):
                try:
                    data = json.loads(response)
                    if "thoughts" in data:
                        return data["thoughts"]
                    elif "reasoning" in data:
                        return data["reasoning"]
                except json.JSONDecodeError:
                    pass

            thought_patterns = [
                r"<thought>(.*?)</thought>",
                r"Thought:\s*(.*?)(?:\n|$)",
                r"Reasoning:\s*(.*?)(?:\n|$)",
                r"Analysis:\s*(.*?)(?:\n|$)",
            ]

            for pattern in thought_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()

            return ""

        except Exception:
            return ""

    def parse_coordinates(self, response: str) -> list[float]:
        """Parse the coordinates from the model response."""
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
        """Parse the reasoning thoughts from the model response."""
        response = response.strip()
        if response.startswith("{") and response.endswith("}"):
            try:
                data = eval(response)
                return data.get("thoughts", "")
            except Exception as e:
                raise ValueError(f"Failed to parse thoughts: {e}")
        raise ValueError("Invalid format for thoughts.")

    def parse_action(self, response: str):
        """Parse the action from the model response.
        
        Returns a tuple of (function_name, args_dict, status) or (None, None, None) if parsing fails.
        For drag operations, args_dict will contain start_coordinate and end_coordinate.
        """
        return None, None, None

    def parse_args(self, response: str) -> dict:
        """Parse the arguments from the model response."""
        response = response.strip()
        if response.startswith("{") and response.endswith("}"):
            try:
                data = eval(response)
                return data.get("args", {})
            except Exception as e:
                raise ValueError(f"Failed to parse args: {e}")
        raise ValueError("Invalid format for args.")

    def parse_status(self, response: str) -> str:
        """Parse the status from the model response."""
        response = response.strip()
        if response.startswith("{") and response.endswith("}"):
            try:
                data = eval(response)
                return data.get("status", "")
            except Exception as e:
                raise ValueError(f"Failed to parse status: {e}")
        raise ValueError("Invalid format for status.")

    def parse_control_list(self, response: str) -> list:
        """Parse the control list from the model response."""
        response = response.strip()
        if response.startswith("[") and response.endswith("]"):
            try:
                controls = eval(response)
                if isinstance(controls, list):
                    return controls
            except Exception as e:
                raise ValueError(f"Failed to parse control list: {e}")
        raise ValueError("Invalid format for control list.")
