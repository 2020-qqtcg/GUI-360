import os
import re
import base64
import json
import json5
import numpy as np
from io import BytesIO
from openai import OpenAI
from PIL import Image
from models.base import BaseModel
from prompts.prompt_action_prediction import ACTION_PREDICTION_USER_PROMPT_QWEN, ACTION_PREDICTION_A11Y_USER_PROMPT_QWEN
from prompts.prompt_grounding import GROUNDING_USER_PROMPT_QWEN
from prompts.prompt_screen_parsing import (
    SCREEN_PARSING_SYS_PROMPT,
    SCREEN_PARSING_USER_PROMPT,
)

# Try to import qwen-vl-utils for smart_resize, fall back to manual implementation if not available
try:
    from qwen_vl_utils import smart_resize
    HAS_QWEN_VL_UTILS = True
except ImportError as e:
    print(f"Error importing qwen-vl-utils: {e}")
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


class Qwen25VL7B(BaseModel):
    def __init__(
        self, api_url: str | None = None, model_name: str = "qwen2.5-vl-7b-instruct", use_smart_resize: bool = True
    ):
        """
        Initialize Qwen 2.5 VL 7B API client for GUI grounding tasks.

        Args:
            api_url: The base API endpoint URL (OpenAI compatible format), e.g. http://localhost:8000/v1
            model_name: Model name to use in API requests
            use_smart_resize: Whether to use smart_resize for better coordinate accuracy
        """
        super().__init__(model_name)
        
        if api_url is None:
            api_port = os.getenv("API_PORT", 8000)
            base_url = f"http://localhost:{api_port}/v1"
        else:
            base_url = str(api_url).rstrip("/")
            if base_url.endswith("/chat/completions"):
                base_url = base_url[: -len("/chat/completions")]
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"

        self.base_url = base_url
        self.use_smart_resize = use_smart_resize
        
        # Store original image dimensions for coordinate transformation
        self.original_dimensions = None
        self.resized_dimensions = None

        # Initialize OpenAI-compatible client with timeout (api_key taken from env as in example)
        self.client = OpenAI(
            api_key=f"{os.getenv('API_KEY', '0')}",
            base_url=self.base_url,
            timeout=600,  # 60秒超时
        )

        # print("Initialized Qwen 2.5 VL 7B API client")
        # print(f"Base URL: {self.base_url}")
        # print(f"Model: {self.model_name}")
        # print(f"Smart resize enabled: {self.use_smart_resize}")
        # if not HAS_QWEN_VL_UTILS and self.use_smart_resize:
        #     print("Warning: qwen-vl-utils not found, using manual smart_resize implementation")

    def _encode_image(self, image_path):
        """
        Encode image to base64 string for API request.

        Args:
            image_path: Path to the image file

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

    def _transform_coordinates_to_original(self, coordinates: list[float]) -> list[float]:
        """
        Transform coordinates from resized image back to original image coordinates.
        
        Args:
            coordinates: [x, y] coordinates from model output (absolute pixel positions on resized image)
            
        Returns:
            list[float]: Transformed coordinates for original image
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

    def predict(self, system_prompt: str, user_prompt: str, image_path: str, max_tokens: int = 2048, temperature: float = 0.0, *args, **kwargs):
        """
        Universal predict method that takes system and user prompts.
        """
        try:
            # Load and preprocess image
            if isinstance(image_path, str) and (
                image_path.startswith("http://") or image_path.startswith("https://")
            ):
                # Download image from URL
                import requests
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
                processed_image = self._preprocess_image_with_smart_resize(image)
                
                # Convert processed image back to base64
                buffer = BytesIO()
                processed_image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_url = f"data:image/png;base64,{base64_image}"
            else:
                # Load local image file
                image = Image.open(image_path)
                processed_image = self._preprocess_image_with_smart_resize(image)
                
                # Convert processed image to base64
                buffer = BytesIO()
                processed_image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_url = f"data:image/png;base64,{base64_image}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]

            # Make API request via OpenAI-compatible client
            result = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=8192,
                temperature=temperature,
            )

            # Parse response
            response_text = result.choices[0].message.content
            return response_text
        except Exception as e:
            print(f"Error during prediction: {e}")
            return ""

    def construct_grounding_prompt(self, thought: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """Construct system and user prompts for grounding task."""
        # Qwen doesn't use separate system prompts, so we return empty system prompt
        system_prompt = ""
        user_prompt = GROUNDING_USER_PROMPT_QWEN.format(
            instruction=thought
        )
        return system_prompt, user_prompt

    def construct_action_prompt(self, instruction: str, history: str, actions: str, resolution: tuple[int, int]) -> tuple[str, str]:
        """Construct system and user prompts for action prediction task."""
        # Qwen doesn't use separate system prompts, so we return empty system prompt
        system_prompt = ""
        
        # Format history properly
        history_formatted = "\n".join(history) if isinstance(history, list) else history if history else ""
        
        user_prompt = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
            instruction=instruction,
            history=history_formatted,
            actions=actions
        )
        return system_prompt, user_prompt

    def construct_action_a11y_prompt(
        self, instruction: str, history: list, actions: str, control_infos: dict
    ) -> tuple[str, str]:
        """Construct system and user prompts for A11Y action prediction task."""
        import json
        
        # Format history as string
        history_str = "\n".join(history) if history else "None"
        
        # Qwen doesn't use separate system prompts, so we return empty system prompt
        system_prompt = ""
        user_prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_QWEN.format(
            instruction=instruction,
            a11y=control_infos,
            history=history_str,
            actions=actions
        )
        
        return system_prompt, user_prompt

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
                try:
                    parsed = json5.loads(response)
                except Exception:
                    def repair_and_parse_json(response: str):
                        """
                        Attempts to parse a JSON string. If parsing fails, it discards
                        the incomplete JSON object at the end and re-parses.

                        Args:
                            response (str): The JSON string to be parsed.

                        Returns:
                            list or None: The parsed Python object (usually a list) if successful,
                                        or None if it fails after all attempts.
                        """
                        # 1. Strip markdown code block markers and leading/trailing whitespace
                        cleaned_response = response.strip().strip('`').strip('json').strip()
                        
                        # 2. Try to parse the complete response directly
                        try:
                            parsed_data = json5.loads(cleaned_response)
                            print("✅ Successfully parsed the complete JSON string.")
                            return parsed_data
                        except Exception:
                            # 3. If direct parsing fails, attempt to repair
                            print("❌ Direct parsing failed. Attempting to repair the incomplete JSON.")
                            
                            # Find the last closing brace '}'
                            last_brace_index = cleaned_response.rfind('}')
                            if last_brace_index != -1:
                                # Slice the string up to the last complete object and add a closing bracket ']'
                                repaired_response = cleaned_response[:last_brace_index + 1] + ']'
                                
                                # 4. Try parsing the repaired string
                                try:
                                    parsed_data = json5.loads(repaired_response)
                                    print("🛠️ Successfully parsed the repaired JSON string.")
                                    return parsed_data
                                except (json5.Json5Error, ValueError, IndexError) as e:
                                    print(f"❌ Failed to parse the repaired string: {e}")
                                    return None
                            else:
                                print("❗ Could not find a complete JSON object to repair.")
                                return None
                    parsed = repair_and_parse_json(response)

                if "bbox_2d" in response and isinstance(parsed, list):
                    for item in parsed:
                        item["control_text"] = item.get("label", "")
                        item["control_rect"] = item.get("bbox_2d", [])
                        del item["bbox_2d"]
                        del item["label"]
                    return parsed

                if "rectangle" in response and isinstance(parsed, list):
                    for item in parsed:
                        item["control_text"] = item.get("label", "") if item.get("label", "") else item.get("text", "")
                        item["control_rect"] = item.get("rectangle", [])

                    return parsed

                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and any(
                    key in parsed for key in ["control_text", "control_rect"]
                ):
                    return [parsed]
                else:
                    print(f"Unexpected response format: {type(parsed)}")
                    return []
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from response: {e}")
                return []

        except Exception as e:
            print(f"Error parsing screen parsing response: {e}")
            return []

    def parse_coordinates(self, response):
        """
        Extract coordinates from model response.

        Args:
            response: Model response text

        Returns:
            List[float]: [x, y] coordinates
        """
        try:
            text = response or ""

            def to_float(value_string: str) -> float:
                s = str(value_string).strip()
                if s.endswith("%"):
                    return float(s[:-1].strip()) / 100.0
                return float(s)

            def try_pair(x_string: str, y_string: str):
                try:
                    return [to_float(x_string), to_float(y_string)]
                except Exception:
                    return None

            def center_of_rectangle(numbers: list[float]) -> list[float] | None:
                if len(numbers) != 4:
                    return None
                x1, y1, x2, y2 = [float(n) for n in numbers]
                # If (left, top, width, height) likely (small widths/heights or keywords handled elsewhere)
                if x2 >= 0 and y2 >= 0 and ((x2 < 1.0) or (y2 < 1.0)):
                    return [x1 + x2 / 2.0, y1 + y2 / 2.0]
                # Otherwise assume (x1, y1, x2, y2)
                return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

            def try_numbers_list(numbers: list[float]) -> list[float] | None:
                if len(numbers) == 2:
                    return [float(numbers[0]), float(numbers[1])]
                if len(numbers) == 4:
                    return center_of_rectangle(numbers)
                return None

            # 1) <coordinate>...</coordinate> and similar tags
            for tag in ["coordinate", "point", "position", "click"]:
                m = re.search(
                    rf"<\s*{tag}[^>]*>(.*?)</\s*{tag}\s*>",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if m:
                    inner = m.group(1)
                    # [x, y]
                    m2 = re.search(r"\[\s*([^\]]+?)\s*\]", inner)
                    if m2:
                        parts = [p.strip() for p in m2.group(1).split(",")]
                        try:
                            nums = [to_float(p) for p in parts if p][:4]
                            pair = try_numbers_list(nums)
                            if pair:
                                # Transform coordinates back to original image dimensions
                                transformed_coordinates = self._transform_coordinates_to_original(pair)
                                return transformed_coordinates
                        except Exception:
                            pass
                    # x=.. y=..
                    m3 = re.search(
                        r"(?i)\bx\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?)\b.*?\by\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?)",
                        inner,
                        re.DOTALL,
                    )
                    if m3:
                        pair = try_pair(m3.group(1), m3.group(2))
                        if pair:
                            # Transform coordinates back to original image dimensions
                            transformed_coordinates = self._transform_coordinates_to_original(pair)
                            return transformed_coordinates

            # 2) <coordinate x=".." y=".."/>
            m = re.search(
                r"<\s*(coordinate|point|position|click)[^>]*\bx\s*=\s*['\"]([^'\"]+)['\"][^>]*\by\s*=\s*['\"]([^'\"]+)['\"][^>]*/?>",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if m:
                pair = try_pair(m.group(2), m.group(3))
                if pair:
                    # Transform coordinates back to original image dimensions
                    transformed_coordinates = self._transform_coordinates_to_original(pair)
                    return transformed_coordinates

            # 3) Rectangle/bbox tags -> center
            for tag in ["rectangle", "bbox", "box", "bounding_box", "region"]:
                m = re.search(
                    rf"<\s*{tag}[^>]*>(.*?)</\s*{tag}\s*>",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if m:
                    inner = m.group(1)
                    m2 = re.search(r"\[\s*([^\]]+?)\s*\]", inner)
                    if m2:
                        parts = [p.strip() for p in m2.group(1).split(",")]
                        try:
                            nums = [to_float(p) for p in parts if p][:4]
                            pair = try_numbers_list(nums)
                            if pair:
                                # Transform coordinates back to original image dimensions
                                transformed_coordinates = self._transform_coordinates_to_original(pair)
                                return transformed_coordinates
                        except Exception:
                            pass
                    named_rect = re.search(
                        r"(?i)(x1|left)\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?).*?(y1|top)\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?).*?(x2|right)\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?).*?(y2|bottom)\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?)",
                        inner,
                        re.DOTALL,
                    )
                    if named_rect:
                        nums = [
                            to_float(named_rect.group(2)),
                            to_float(named_rect.group(4)),
                            to_float(named_rect.group(6)),
                            to_float(named_rect.group(8)),
                        ]
                        pair = try_numbers_list(nums)
                        if pair:
                            # Transform coordinates back to original image dimensions
                            transformed_coordinates = self._transform_coordinates_to_original(pair)
                            return transformed_coordinates
                    ltrb = re.search(
                        r"(?i)left\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?).*?top\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?).*?width\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?).*?height\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?)",
                        inner,
                        re.DOTALL,
                    )
                    if ltrb:
                        left, top, width, height = [
                            to_float(ltrb.group(i)) for i in range(1, 5)
                        ]
                        pair = [left + width / 2.0, top + height / 2.0]
                        # Transform coordinates back to original image dimensions
                        transformed_coordinates = self._transform_coordinates_to_original(pair)
                        return transformed_coordinates

            # 4) JSON-like blocks
            for jc in re.findall(r"\{[^{}]*\}", text, re.DOTALL):
                try:
                    obj = json.loads(jc)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    if (
                        "coordinate" in obj
                        and isinstance(obj["coordinate"], (list, tuple))
                        and len(obj["coordinate"]) >= 2
                    ):
                        pair = try_pair(
                            str(obj["coordinate"][0]), str(obj["coordinate"][1])
                        )
                        if pair:
                            # Transform coordinates back to original image dimensions
                            transformed_coordinates = self._transform_coordinates_to_original(pair)
                            return transformed_coordinates
                    if "x" in obj and "y" in obj:
                        pair = try_pair(str(obj["x"]), str(obj["y"]))
                        if pair:
                            # Transform coordinates back to original image dimensions
                            transformed_coordinates = self._transform_coordinates_to_original(pair)
                            return transformed_coordinates
                    for key in ("point", "position", "click", "center"):
                        if (
                            key in obj
                            and isinstance(obj[key], dict)
                            and "x" in obj[key]
                            and "y" in obj[key]
                        ):
                            pair = try_pair(str(obj[key]["x"]), str(obj[key]["y"]))
                            if pair:
                                # Transform coordinates back to original image dimensions
                                transformed_coordinates = self._transform_coordinates_to_original(pair)
                                return transformed_coordinates
                    for key in ("rectangle", "bbox", "box", "region"):
                        if (
                            key in obj
                            and isinstance(obj[key], (list, tuple))
                            and len(obj[key]) >= 4
                        ):
                            nums = [to_float(str(n)) for n in obj[key][:4]]
                            pair = try_numbers_list(nums)
                            if pair:
                                # Transform coordinates back to original image dimensions
                                transformed_coordinates = self._transform_coordinates_to_original(pair)
                                return transformed_coordinates

            # 5) Labeled x/y in free text
            m = re.search(
                r"(?i)\bx\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?)\b.*?\by\s*[:=]\s*([-+]?\d+(?:\.\d+)?%?)",
                text,
                re.DOTALL,
            )
            if m:
                pair = try_pair(m.group(1), m.group(2))
                if pair:
                    # Transform coordinates back to original image dimensions
                    transformed_coordinates = self._transform_coordinates_to_original(pair)
                    return transformed_coordinates

            # 6) Keyword followed by pair in () or []
            m = re.search(
                r"(?i)(final|answer|click|target|coordinate|position|point|center)[^\d\-]*([\(\[]\s*[-+]?\d+(?:\.\d+)?%?\s*[,; ]\s*[-+]?\d+(?:\.\d+)?%?\s*[\)\]])",
                text,
            )
            if m:
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?%?", m.group(2))
                if len(nums) >= 2:
                    pair = try_pair(nums[0], nums[1])
                    if pair:
                        # Transform coordinates back to original image dimensions
                        transformed_coordinates = self._transform_coordinates_to_original(pair)
                        return transformed_coordinates

            # 7) Any [...] arrays; prefer 4-number rectangles nearby bbox keywords; else 2-number pairs
            arrays = list(re.finditer(r"\[\s*([^\]]+?)\s*\]", text))
            fallback_pair = None
            for arr in arrays:
                content = arr.group(1)
                parts = [p.strip() for p in content.split(",") if p.strip()]
                try:
                    nums = [to_float(p) for p in parts]
                except Exception:
                    continue
                if len(nums) == 2:
                    fallback_pair = [float(nums[0]), float(nums[1])]
                elif len(nums) == 4:
                    vicinity = text[
                        max(0, arr.start() - 80) : min(len(text), arr.end() + 80)
                    ].lower()
                    if any(
                        k in vicinity
                        for k in [
                            "rect",
                            "bbox",
                            "box",
                            "rectangle",
                            "region",
                            "bound",
                            "left",
                            "top",
                            "right",
                            "bottom",
                            "width",
                            "height",
                        ]
                    ):
                        pair = try_numbers_list(nums)
                        if pair:
                            # Transform coordinates back to original image dimensions
                            transformed_coordinates = self._transform_coordinates_to_original(pair)
                            return transformed_coordinates
                    pair = try_numbers_list(nums)
                    if pair:
                        fallback_pair = pair
            if fallback_pair:
                # Transform coordinates back to original image dimensions
                transformed_coordinates = self._transform_coordinates_to_original(fallback_pair)
                return transformed_coordinates

            # 8) Parentheses pairs like (x, y)
            m = re.search(
                r"\(\s*([-+]?\d+(?:\.\d+)?%?)\s*[, ]\s*([-+]?\d+(?:\.\d+)?%?)\s*\)",
                text,
            )
            if m:
                pair = try_pair(m.group(1), m.group(2))
                if pair:
                    # Transform coordinates back to original image dimensions
                    transformed_coordinates = self._transform_coordinates_to_original(pair)
                    return transformed_coordinates

            # 9) Loose numbers after keywords
            m = re.search(
                r"(?i)(final|answer|click|target|coordinate|position|point|center)[^\d\-]*([-+]?\d+(?:\.\d+)?%?)\s*[,; ]\s*([-+]?\d+(?:\.\d+)?%?)",
                text,
            )
            if m:
                pair = try_pair(m.group(2), m.group(3))
                if pair:
                    # Transform coordinates back to original image dimensions
                    transformed_coordinates = self._transform_coordinates_to_original(pair)
                    return transformed_coordinates

            print(f"Could not extract coordinates from response: {response}")
            coordinates = [0.0, 0.0]

        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            coordinates = [0.0, 0.0]
        
        # Transform coordinates back to original image dimensions
        transformed_coordinates = self._transform_coordinates_to_original(coordinates)
        return transformed_coordinates

    def parse_action(self, response):
        """
        Extract action information from model response with A11Y support.
        
        Expected format:
        <tool_call>
        {
            "function": "type",
            "args": {"text": "=ATAN(1)", "clear_current_text": true},
            "status": "CONTINUE"
        }
        </tool_call>
        
        For drag operations:
        <tool_call>
        {
            "function": "drag",
            "args": {"start_coordinate": [100, 100], "end_coordinate": [200, 200], "button": "left"},
            "status": "CONTINUE"
        }
        </tool_call>
        
        For A11Y operations with control_label:
        <tool_call>
        {
            "function": "click",
            "args": {"control_label": 15, "coordinate": null, "button": "left"},
            "status": "CONTINUE"
        }
        </tool_call>
        
        Returns:
            Tuple of (function_name, args_dict, status) or (None, None, None) if parsing fails
        """
        try:
            # Look for tool_call block with more flexible args parsing (allow empty function name)
            pattern = r'<tool_call>\s*\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}\s*</tool_call>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            
            if match:
                function_name = match.group(1)
                args_str = match.group(2)
                status = match.group(3)
                
                # Handle OVERALL_FINISH case where function can be empty
                if status == "OVERALL_FINISH" and function_name == "":
                    try:
                        args_dict = json.loads(args_str)
                        return function_name, args_dict, status
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse args JSON: {args_str}, error: {e}")
                        return function_name, {}, status
                
                # For other cases, function_name must be non-empty
                if function_name is not None:
                    try:
                        args_dict = json.loads(args_str)
                        # Validate and transform coordinates for drag operation
                        if function_name == "drag":
                            if "start_coordinate" not in args_dict or "end_coordinate" not in args_dict:
                                print(f"Warning: drag operation missing required coordinates: {args_dict}")
                            # Ensure coordinates are lists of two numbers and transform them
                            if "start_coordinate" in args_dict:
                                start = args_dict["start_coordinate"]
                                if isinstance(start, list) and len(start) == 2:
                                    args_dict["start_coordinate"] = self._transform_coordinates_to_original(start)
                                else:
                                    print(f"Warning: invalid start_coordinate format: {start}")
                            if "end_coordinate" in args_dict:
                                end = args_dict["end_coordinate"]
                                if isinstance(end, list) and len(end) == 2:
                                    args_dict["end_coordinate"] = self._transform_coordinates_to_original(end)
                                else:
                                    print(f"Warning: invalid end_coordinate format: {end}")
                        
                        # Transform other coordinate fields
                        if "coordinate" in args_dict and isinstance(args_dict["coordinate"], list) and len(args_dict["coordinate"]) >= 2:
                            args_dict["coordinate"] = self._transform_coordinates_to_original(args_dict["coordinate"])
                        
                        return function_name, args_dict, status
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse args JSON: {args_str}, error: {e}")
                        return function_name, {}, status
            
            # Alternative pattern without tool_call tags (allow empty function name)
            pattern2 = r'\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}'
            match2 = re.search(pattern2, response, re.DOTALL)
            
            if match2:
                function_name = match2.group(1)
                args_str = match2.group(2)
                status = match2.group(3)
                
                # Handle OVERALL_FINISH case where function can be empty
                if status == "OVERALL_FINISH" and function_name == "":
                    try:
                        args_dict = json.loads(args_str)
                        return function_name, args_dict, status
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse args JSON: {args_str}, error: {e}")
                        return function_name, {}, status
                
                # For other cases, function_name must be non-empty
                if function_name is not None:
                    try:
                        args_dict = json.loads(args_str)
                        # Validate and transform coordinates for drag operation
                        if function_name == "drag":
                            if "start_coordinate" not in args_dict or "end_coordinate" not in args_dict:
                                print(f"Warning: drag operation missing required coordinates: {args_dict}")
                            # Transform coordinates
                            if "start_coordinate" in args_dict and isinstance(args_dict["start_coordinate"], list) and len(args_dict["start_coordinate"]) >= 2:
                                args_dict["start_coordinate"] = self._transform_coordinates_to_original(args_dict["start_coordinate"])
                            if "end_coordinate" in args_dict and isinstance(args_dict["end_coordinate"], list) and len(args_dict["end_coordinate"]) >= 2:
                                args_dict["end_coordinate"] = self._transform_coordinates_to_original(args_dict["end_coordinate"])
                        
                        # Transform other coordinate fields
                        if "coordinate" in args_dict and isinstance(args_dict["coordinate"], list) and len(args_dict["coordinate"]) >= 2:
                            args_dict["coordinate"] = self._transform_coordinates_to_original(args_dict["coordinate"])
                        
                        return function_name, args_dict, status
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse args JSON: {args_str}, error: {e}")
                        return function_name, {}, status
            
            # Try to parse JSON blocks that might contain action information
            json_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            for json_block in json_blocks:
                try:
                    data = json.loads(json_block)
                    if isinstance(data, dict):
                        # Check if it has the expected action structure
                        if "function" in data:
                            function_name = data.get("function")
                            args_dict = data.get("args", {})
                            status = data.get("status", "CONTINUE")
                            
                            # Handle OVERALL_FINISH case where function can be empty
                            if status == "OVERALL_FINISH" and function_name == "":
                                return function_name, args_dict, status
                            
                            # For other cases, function_name must be non-empty
                            if function_name is not None:
                                # Validate and transform coordinates for drag operation
                                if function_name == "drag":
                                    if "start_coordinate" not in args_dict or "end_coordinate" not in args_dict:
                                        print(f"Warning: drag operation missing required coordinates: {args_dict}")
                                    # Transform coordinates
                                    if "start_coordinate" in args_dict and isinstance(args_dict["start_coordinate"], list) and len(args_dict["start_coordinate"]) >= 2:
                                        args_dict["start_coordinate"] = self._transform_coordinates_to_original(args_dict["start_coordinate"])
                                    if "end_coordinate" in args_dict and isinstance(args_dict["end_coordinate"], list) and len(args_dict["end_coordinate"]) >= 2:
                                        args_dict["end_coordinate"] = self._transform_coordinates_to_original(args_dict["end_coordinate"])
                                
                                # Transform other coordinate fields
                                if "coordinate" in args_dict and isinstance(args_dict["coordinate"], list) and len(args_dict["coordinate"]) >= 2:
                                    args_dict["coordinate"] = self._transform_coordinates_to_original(args_dict["coordinate"])
                                
                                return function_name, args_dict, status
                        
                        # Check for nested tool_call structure
                        if "tool_call" in data and isinstance(data["tool_call"], dict):
                            tool_call = data["tool_call"]
                            function_name = tool_call.get("function")
                            args_dict = tool_call.get("args", {})
                            status = tool_call.get("status", "CONTINUE")
                            
                            # Handle OVERALL_FINISH case where function can be empty
                            if status == "OVERALL_FINISH" and function_name == "":
                                return function_name, args_dict, status
                            
                            # For other cases, function_name must be non-empty
                            if function_name is not None:
                                # Validate and transform coordinates for drag operation
                                if function_name == "drag":
                                    if "start_coordinate" not in args_dict or "end_coordinate" not in args_dict:
                                        print(f"Warning: drag operation missing required coordinates: {args_dict}")
                                    # Transform coordinates
                                    if "start_coordinate" in args_dict and isinstance(args_dict["start_coordinate"], list) and len(args_dict["start_coordinate"]) >= 2:
                                        args_dict["start_coordinate"] = self._transform_coordinates_to_original(args_dict["start_coordinate"])
                                    if "end_coordinate" in args_dict and isinstance(args_dict["end_coordinate"], list) and len(args_dict["end_coordinate"]) >= 2:
                                        args_dict["end_coordinate"] = self._transform_coordinates_to_original(args_dict["end_coordinate"])
                                
                                # Transform other coordinate fields
                                if "coordinate" in args_dict and isinstance(args_dict["coordinate"], list) and len(args_dict["coordinate"]) >= 2:
                                    args_dict["coordinate"] = self._transform_coordinates_to_original(args_dict["coordinate"])
                                
                                return function_name, args_dict, status
                except json.JSONDecodeError:
                    continue
            
            # Try to extract from text patterns
            # Look for function: "name" pattern
            func_match = re.search(r'"function":\s*"([^"]+)"', response)
            args_match = re.search(r'"args":\s*(\{.*?\})', response, re.DOTALL)
            status_match = re.search(r'"status":\s*"([^"]+)"', response)
            
            if func_match:
                function_name = func_match.group(1)
                args_dict = {}
                status = "CONTINUE"
                
                if args_match:
                    try:
                        args_dict = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        args_dict = {}
                
                if status_match:
                    status = status_match.group(1)
                
                # Validate and transform coordinates for drag operation
                if function_name == "drag":
                    if "start_coordinate" not in args_dict or "end_coordinate" not in args_dict:
                        print(f"Warning: drag operation missing required coordinates: {args_dict}")
                    # Transform coordinates
                    if "start_coordinate" in args_dict and isinstance(args_dict["start_coordinate"], list) and len(args_dict["start_coordinate"]) >= 2:
                        args_dict["start_coordinate"] = self._transform_coordinates_to_original(args_dict["start_coordinate"])
                    if "end_coordinate" in args_dict and isinstance(args_dict["end_coordinate"], list) and len(args_dict["end_coordinate"]) >= 2:
                        args_dict["end_coordinate"] = self._transform_coordinates_to_original(args_dict["end_coordinate"])
                
                # Transform other coordinate fields
                if "coordinate" in args_dict and isinstance(args_dict["coordinate"], list) and len(args_dict["coordinate"]) >= 2:
                    args_dict["coordinate"] = self._transform_coordinates_to_original(args_dict["coordinate"])
                
                return function_name, args_dict, status
            
            print(f"Could not extract action from response: {response[:200]}...")
            return None, None, None
            
        except Exception as e:
            print(f"Error parsing action from response: {e}")
            return None, None, None

    def parse_thoughts(self, response):
        """
        Extract thoughts from model response.

        Args:
            response: Model response text

        Returns:
            str: Thoughts
        """
        return response


# For backward compatibility and easy import
def load_model(api_url: str | None = None, model_name: str = "qwen2.5-vl-7b-instruct", use_smart_resize: bool = True):
    """
    Load and return a Qwen 2.5 VL 7B API client instance.

    Args:
        api_url: The API endpoint URL (OpenAI compatible format)
        model_name: Model name to use in API requests
        use_smart_resize: Whether to use smart_resize for better coordinate accuracy

    Returns:
        Qwen25VL7B: API client instance
    """
    return Qwen25VL7B(api_url, model_name, use_smart_resize)


# Example usage
if __name__ == "__main__":
    # Test the API client with vLLM deployment (same URL pattern as example)
    api_port = os.getenv("API_PORT", 8000)
    api_url = f"http://localhost:{api_port}/v1"
    model = Qwen25VL7B(api_url)

    # Example prediction
    test_instruction = "To proceed, I need to accept Excel's correction by clicking the 'Yes' button in the dialog. This will fix the formula to '=ATAN(1)' in cell H4 and complete the ATAN calculation as required by the sub-task."
    test_image = "data/test/image/word/qabench/success/word_1_1/action_step1.png"

    # Note: This would require an actual image file and running vLLM server to test
    # coordinates = model.predict(test_instruction, test_image)
    # print(f"Predicted coordinates: {coordinates}")
