import json
import logging
import base64
import os

import json5
from openai import OpenAI

from models.base import BaseModel
from config import EvaluationConfig
from prompts.prompt_action_prediction import (
    ACTION_PREDICTION_SYS_PROMPT_GPT,
    ACTION_PREDICTION_USER_PROMPT_GPT,
    ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT,
    ACTION_PREDICTION_A11Y_USER_PROMPT_GPT,
)
from prompts.prompt_screen_parsing import (
    SCREEN_PARSING_SYS_PROMPT,
    SCREEN_PARSING_USER_PROMPT,
)
from prompts.prompt_grounding import GROUNDING_SYS_PROMPT, GROUNDING_USER_PROMPT


def encode_image(image_path: str) -> str:
    """Encode image to base64 data URL for OpenAI API."""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_image}"


class GPT_predictor(BaseModel):
    def __init__(self, model_name: str, config: EvaluationConfig = None):
        """Initialize the GPT predictor with the model name."""
        super().__init__(model_name)
        
        # Use config if provided, otherwise use environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config.")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """Universal predict method that takes system and user prompts."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image(image_path)},
                        },
                    ],
                },
            ]

            import time
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages,
                temperature=temperature
            )
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return ""

    def construct_grounding_prompt(
        self, thought: str, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for grounding task."""
        system_prompt = GROUNDING_SYS_PROMPT
        user_prompt = GROUNDING_USER_PROMPT.format(
            instruction=thought, resolution=resolution
        )
        return system_prompt, user_prompt

    def construct_action_prompt(
        self, instruction: str, history: str, actions: str, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for action prediction task."""
        system_prompt = ACTION_PREDICTION_SYS_PROMPT_GPT
        user_prompt = ACTION_PREDICTION_USER_PROMPT_GPT.format(
            instruction=instruction,
            resolution=resolution,
            history=history,
            actions=actions,
        )
        return system_prompt, user_prompt

    def construct_action_a11y_prompt(
        self, instruction: str, history: list, actions: str, control_infos: dict
    ) -> tuple[str, str]:
        """Construct system and user prompts for A11Y action prediction task."""
        import json

        # Format history as string
        history_str = "\n".join(history) if history else "None"

        # Use GPT-specific A11Y prompts
        system_prompt = ACTION_PREDICTION_A11Y_SYS_PROMPT_GPT
        user_prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_GPT.format(
            instruction=instruction,
            a11y=json.dumps(control_infos, indent=2),
            history=history_str,
            actions=actions,
        )

        return system_prompt, user_prompt

    def parse_coordinates(self, response: str) -> list[float]:
        """Parse the coordinates from the GPT response."""
        response = response.strip()

        if response.startswith("```json"):
            response = response.split("```json")[1].split("```")[0]
            response = response.strip()

        try:
            data = json5.loads(response)
            coordinates = data.get("coordinates", [])
            if not coordinates:
                coordinates = [0.0, 0.0]
                self.logger.warning("No coordinates found in response, using default.")
            return coordinates
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse coordinates from response: {response}")
            return [0.0, 0.0]

    def parse_thoughts(self, response: str) -> str:
        """Parse the reasoning thoughts from the GPT response."""
        response = response.strip()

        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
            response = response.strip()

        try:
            data = json5.loads(response)
            return data.get("thoughts", "")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse thoughts from response: {response}")

    def parse_action(self, response):
        """
        Extract action information from GPT response.

        Expected format:
        ```json
        {
            "thoughts": "...",
            "tool_call": {
                "function": "type",
                "args": {"coordinate": [100, 100], "keys": "Hello"},
                "status": "CONTINUE"
            }
        }
        ```

        Returns:
            Tuple of (function_name, args_dict, status) or (None, None, None) if parsing fails
        """
        try:
            import re

            response = response.strip()

            def extract_tool_call(data):
                if isinstance(data, dict) and "tool_call" in data:
                    tool_call = data["tool_call"]
                    if isinstance(tool_call, dict):
                        function_name = tool_call.get("function")
                        args_dict = tool_call.get("args", {})
                        status = tool_call.get("status", "CONTINUE")

                        if status == "OVERALL_FINISH" and function_name == "":
                            return function_name, args_dict, status

                        if function_name is not None:
                            if function_name == "drag":
                                if (
                                    "start_coordinate" not in args_dict
                                    or "end_coordinate" not in args_dict
                                ):
                                    self.logger.warning(
                                        f"drag operation missing required coordinates: {args_dict}"
                                    )
                            return function_name, args_dict, status
                return None, None, None

            json_match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                try:
                    data = json5.loads(json_content)
                    function_name, args_dict, status = extract_tool_call(data)
                    if function_name is not None:
                        return function_name, args_dict, status
                except json.JSONDecodeError:
                    pass

            try:
                data = json5.loads(response)
                function_name, args_dict, status = extract_tool_call(data)
                if function_name is not None:
                    return function_name, args_dict, status
            except json.JSONDecodeError:
                pass

            json_pattern = (
                r'\{[^{}]*"tool_call"[^{}]*\{[^{}]*"function"[^{}]*\}[^{}]*\}'
            )
            json_matches = re.finditer(json_pattern, response, re.DOTALL)
            for match in json_matches:
                json_str = match.group(0)
                try:
                    brace_count = 0
                    start_pos = match.start()
                    end_pos = start_pos

                    for i, char in enumerate(response[start_pos:]):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = start_pos + i + 1
                                break

                    if brace_count == 0:
                        json_str = response[start_pos:end_pos]
                        data = json5.loads(json_str)
                        function_name, args_dict, status = extract_tool_call(data)
                        if function_name is not None:
                            return function_name, args_dict, status
                except (json.JSONDecodeError, IndexError):
                    continue

            # Look for tool_call block with more flexible args parsing
            pattern = r'<tool_call>\s*\{\s*"function":\s*"([^"]+)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}\s*</tool_call>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

            if match:
                function_name = match.group(1)
                args_str = match.group(2)
                status = match.group(3)

                try:
                    args_dict = json.loads(args_str)
                    if function_name == "drag":
                        if (
                            "start_coordinate" not in args_dict
                            or "end_coordinate" not in args_dict
                        ):
                            self.logger.warning(
                                f"drag operation missing required coordinates: {args_dict}"
                            )
                        if "start_coordinate" in args_dict:
                            start = args_dict["start_coordinate"]
                            if not isinstance(start, list) or len(start) != 2:
                                self.logger.warning(
                                    f"invalid start_coordinate format: {start}"
                                )
                        if "end_coordinate" in args_dict:
                            end = args_dict["end_coordinate"]
                            if not isinstance(end, list) or len(end) != 2:
                                self.logger.warning(
                                    f"invalid end_coordinate format: {end}"
                                )
                    return function_name, args_dict, status
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Failed to parse args JSON: {args_str}, error: {e}"
                    )
                    return function_name, {}, status

            self.logger.warning(
                f"Could not extract action from response: {response[:200]}..."
            )
            return None, None, None

        except Exception as e:
            self.logger.error(f"Error parsing action from response: {e}")
            return None, None, None

    def construct_screen_parsing_prompt(
        self, resolution: tuple[int, int]
    ) -> tuple[str, str]:
        """Construct system and user prompts for screen parsing task.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
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
            response = response.strip()

            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
                response = response.strip()
            elif "```" in response:
                # Handle cases where JSON is in code blocks without json marker
                parts = response.split("```")
                if len(parts) >= 3:
                    response = parts[1].strip()

            # Try to parse as JSON
            try:
                parsed = json5.loads(response)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and any(
                    key in parsed for key in ["control_text", "control_rect"]
                ):
                    return [parsed]
                else:
                    self.logger.warning(f"Unexpected response format: {type(parsed)}")
                    return []
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from response: {e}")
                return []

        except Exception as e:
            self.logger.error(f"Error parsing screen parsing response: {e}")
            return []


# if __name__ == "__main__":

#     # Example usage
#     predictor = GPT_predictor(model_name="gpt-5-20250807")
#     thought = "Find the coordinates of the button."
#     image_path = (
#         "D:\\gui_data\\image\\word\\qabench\\success\\word_1_1\\action_step1.png"
#     )
#     resolution = (1920, 1080)

#     try:
#         coords = predictor.predict(thought, image_path, resolution)
#         print(f"Predicted coordinates: {coords}")
#     except Exception as e:
#         print(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Test with GPT-4 model
    gpt = GPT_predictor(model_name="gpt-4o")
    
    response = gpt.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    )

    response_text = response.choices[0].message.content
    print(f"Response Text: {response_text}")
    print(f"Response: {response}")