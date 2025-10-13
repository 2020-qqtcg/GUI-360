import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json5
from PIL import Image

from config import EvaluationConfig
from evaluator.base import BaseEvaluator, EvaluationResult
from evaluator.tool_definitions import compare_normalized_args, normalize_tool_args_a11y
from prompts.prompt_action_prediction import (
    SUPPORTED_ACTIONS_EXCEL_A11Y,
    SUPPORTED_ACTIONS_PPT_A11Y,
    SUPPORTED_ACTIONS_WORD_A11Y,
)


@dataclass
class ActionPredictionA11yEvaluationResult(EvaluationResult):
    """Result of a single action prediction evaluation with A11Y support."""

    predicted_function: Optional[str] = None
    predicted_args: Optional[Dict[str, Any]] = None
    predicted_status: Optional[str] = None
    ground_truth_function: Optional[str] = None
    ground_truth_args: Optional[Dict[str, Any]] = None
    ground_truth_status: Optional[str] = None
    function_match: bool = False
    args_match: bool = False
    status_match: bool = False
    thoughts: Optional[str] = None


class ActionPredictionA11yEvaluator(BaseEvaluator):
    """Evaluator for GUI action prediction tasks with A11Y support."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        self.results: List[ActionPredictionA11yEvaluationResult] = []

    def clean_a11y(self, a11y):
        """
        Transform a11y information to a more readable format.
        """
        if "application_windows_info" in a11y:
            del a11y["application_windows_info"]["control_rect"]
            del a11y["application_windows_info"]["source"]

        for control_info in a11y["uia_controls_info"]:
            del control_info["control_rect"]
            del control_info["source"]
        
        return a11y

    def load_data(self, task_type):
        """Load data from dataset directory and construct previous actions for each sample."""
        root_path = os.path.join(self.config.root_dir)
        sample_count = 0
        all_samples = []

        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Dataset directory not found: {root_path}")

        data_path = os.path.join(root_path, "data")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        self.logger.info(f"Loading data from: {root_path}")

        # Process /data directory
        domain_folders = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]
        self.logger.info(f"Found {len(domain_folders)} domains: {domain_folders}")

        for domain in domain_folders:
            domain_path = os.path.join(data_path, domain)
            category_folders = [
                c
                for c in os.listdir(domain_path)
                if os.path.isdir(os.path.join(domain_path, c))
            ]

            for category in category_folders:
                category_path = os.path.join(domain_path, category, "success")
                if not os.path.exists(category_path):
                    continue

                jsonl_files = [
                    f for f in os.listdir(category_path) if f.endswith(".jsonl")
                ]
                self.logger.debug(
                    f"Processing {len(jsonl_files)} files in {domain}/{category}"
                )

                for jsonl_file in jsonl_files:
                    file_path = os.path.join(category_path, jsonl_file)

                    try:
                        # Load all steps from the file to construct previous actions
                        all_steps = []
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line_num, line in enumerate(f, 1):
                                if not line.strip():
                                    continue

                                try:
                                    data = json.loads(line.strip())
                                    all_steps.append(
                                        {"line_num": line_num, "data": data}
                                    )
                                except Exception as e:
                                    self.logger.error(
                                        f"Error in {file_path}:{line_num} - {e}"
                                    )
                                    continue

                        # Process each step and construct previous actions
                        for i, step_info in enumerate(all_steps):
                            line_num = step_info["line_num"]
                            data = step_info["data"]

                            # Create sample ID
                            sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{line_num}"

                            # Build image paths - use annotated image for A11Y
                            clean_img_path = os.path.join(
                                root_path,
                                "image",
                                domain,
                                category,
                                data["step"]["screenshot_clean"],
                            )
                            annotated_img_path = os.path.join(
                                root_path,
                                "image",
                                domain,
                                category,
                                data["step"]["screenshot_annotated"],
                            )

                            # Validate annotated image exists (primary for A11Y)
                            if not os.path.exists(annotated_img_path):
                                self.logger.warning(
                                    f"Annotated image not found: {annotated_img_path}"
                                )
                                continue

                            # Check if this sample has the required task type
                            if "action_prediction" in data["step"]["tags"]:
                                # Construct previous actions from earlier steps
                                previous_actions = []
                                for j in range(i):
                                    prev_step_data = all_steps[j]["data"]
                                    prev_thought = prev_step_data["step"]["thought"]
                                    previous_actions.append(
                                        f"Step {j+1}: {prev_thought}"
                                    )

                                # Create sample with previous actions
                                status = data["step"]["status"]
                                if status == "OVERALL_FINISH":
                                    status = "FINISH"
                                elif status == "FINISH":
                                    status = "CONTINUE"

                                control_infos = self.clean_a11y(data["step"].get("control_infos", {}))

                                sample = {
                                    "sample_id": sample_id,
                                    "request": data["request"],
                                    "screenshot_clean": clean_img_path,
                                    "screenshot_annotated": annotated_img_path,
                                    "thought": data["step"]["thought"],
                                    "action": data["step"]["action"],
                                    "status": status,
                                    "domain": domain,
                                    "category": category,
                                    "previous_actions": previous_actions,
                                    "step_index": i + 1,
                                    # Add control_infos for A11Y support
                                    "control_infos": control_infos
                                }

                                if sample["action"].get(
                                    "function", ""
                                ) != "drag" and sample["action"].get("rectangle", {}):
                                    yield sample
                                    sample_count += 1
                                    if sample_count % 1000 == 0:
                                        self.logger.info(
                                            f"Loaded {sample_count} samples so far"
                                        )

                                if (
                                    self.config.max_samples
                                    and sample_count >= self.config.max_samples
                                ):
                                    self.logger.info(
                                        f"Reached max samples limit: {self.config.max_samples}"
                                    )
                                    return

                    except Exception as e:
                        self.logger.error(f"Error reading file {file_path}: {e}")
                        continue

        self.logger.info(f"Loaded {sample_count} samples total")

    def load_previous_results(
        self, results_file_path: str
    ) -> Tuple[List[ActionPredictionA11yEvaluationResult], List[str]]:
        """Load previous evaluation results and extract error cases.

        Returns:
            Tuple of (all_previous_results, error_sample_ids)
        """
        try:
            with open(results_file_path, "r", encoding="utf-8") as f:
                data = json5.load(f)

            previous_results = []
            error_sample_ids = []

            if "detailed_results" in data:
                for result_data in data["detailed_results"]:
                    # Reconstruct ActionPredictionA11yEvaluationResult objects
                    result = ActionPredictionA11yEvaluationResult(
                        success=result_data.get("success", False),
                        predicted_function=result_data.get("predicted_function"),
                        predicted_args=result_data.get("predicted_args", {}),
                        predicted_status=result_data.get("predicted_status"),
                        ground_truth_function=result_data.get("ground_truth_function"),
                        ground_truth_args=result_data.get("ground_truth_args", {}),
                        ground_truth_status=result_data.get("ground_truth_status"),
                        function_match=result_data.get("function_match", False),
                        args_match=result_data.get("args_match", False),
                        status_match=result_data.get("status_match", False),
                        ground_truth_rect=result_data.get("ground_truth_rect", {}),
                        sample_id=result_data.get("sample_id", ""),
                        execution_time=result_data.get("execution_time", 0.0),
                        error_message=result_data.get("error_message"),
                        thoughts=result_data.get("thoughts"),
                        raw_model_output=result_data.get("raw_model_output"),
                    )
                    previous_results.append(result)

                    # Collect error cases (failed or had error messages)
                    if not result.success or result.error_message:
                        error_sample_ids.append(result.sample_id)

            self.logger.info(f"Loaded {len(previous_results)} previous results")
            self.logger.info(f"Found {len(error_sample_ids)} error cases to retry")

            return previous_results, error_sample_ids

        except Exception as e:
            raise RuntimeError(
                f"Failed to load previous results from {results_file_path}: {e}"
            )

    def filter_samples_for_retry(
        self, all_samples: List[Dict[str, Any]], error_sample_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter samples to only include those that need to be retried."""
        retry_samples = []

        for sample in all_samples:
            if sample["sample_id"] in error_sample_ids:
                retry_samples.append(sample)

        self.logger.info(
            f"Filtered {len(retry_samples)} samples for retry out of {len(all_samples)} total samples"
        )
        return retry_samples

    def compare_actions(
        self,
        pred_function: Optional[str],
        pred_args: Optional[Dict],
        pred_status: Optional[str],
        gt_function: str,
        gt_args: Dict,
        gt_status: str,
        gt_rect: Dict = None,
        gt_rect_end: Dict = None,
    ) -> Tuple[bool, bool, bool]:
        """Compare predicted action with ground truth action.
        
        Enhanced for A11Y support - prioritizes control_label matching over coordinate matching.

        Returns:
            Tuple of (function_match, args_match, status_match)
        """
        # Function match
        function_match = (
            pred_function == gt_function if pred_function is not None else False
        )

        # Status match
        status_match = pred_status == gt_status if pred_status else False

        # Args match - enhanced for A11Y control_label support
        args_match = False
        if pred_args is not None and gt_args is not None:
            if pred_function == "drag" and gt_function == "drag":
                # Special comparison for drag operations with rectangle-based matching
                args_match = self._compare_drag_args(
                    pred_args, gt_args, gt_rect, gt_rect_end
                )
            else:
                # Enhanced comparison for A11Y operations
                args_match = self._compare_a11y_args(
                    pred_args, gt_args, gt_rect, pred_function, gt_function
                )

        return function_match, args_match, status_match

    def _compare_drag_args(
        self,
        pred_args: Dict,
        gt_args: Dict,
        gt_rect: Dict = None,
        gt_rect_end: Dict = None,
    ) -> bool:
        """Compare drag operation arguments with rectangle-based matching."""
        try:
            # Normalize both argument sets with A11Y default values
            pred_normalized = normalize_tool_args_a11y("drag", pred_args)
            gt_normalized = normalize_tool_args_a11y("drag", gt_args)

            # Check if both have required drag coordinates
            if (
                "start_coordinate" not in pred_normalized
                or "end_coordinate" not in pred_normalized
            ):
                return False
            if (
                "start_coordinate" not in gt_normalized
                or "end_coordinate" not in gt_normalized
            ):
                return False

            pred_start = pred_normalized["start_coordinate"]
            pred_end = pred_normalized["end_coordinate"]
            gt_start = gt_normalized["start_coordinate"]
            gt_end = gt_normalized["end_coordinate"]

            # Ensure coordinates are lists/tuples with 2 elements
            if not (isinstance(pred_start, (list, tuple)) and len(pred_start) == 2):
                return False
            if not (isinstance(pred_end, (list, tuple)) and len(pred_end) == 2):
                return False
            if not (isinstance(gt_start, (list, tuple)) and len(gt_start) == 2):
                return False
            if not (isinstance(gt_end, (list, tuple)) and len(gt_end) == 2):
                return False

            # Check if predicted coordinates fall within ground truth rectangles
            start_match = True
            end_match = True

            if gt_rect:
                # Check if predicted start coordinate is within ground truth start rectangle
                pred_start_x, pred_start_y = float(pred_start[0]), float(pred_start[1])
                start_match = (
                    gt_rect["left"] <= pred_start_x <= gt_rect["right"]
                    and gt_rect["top"] <= pred_start_y <= gt_rect["bottom"]
                )
            else:
                # Fall back to tolerance-based comparison if no rectangle provided
                tolerance = 25.0  # Default tolerance
                start_match = (
                    abs(float(pred_start[0]) - float(gt_start[0])) <= tolerance
                    and abs(float(pred_start[1]) - float(gt_start[1])) <= tolerance
                )

            if gt_rect_end:
                # Check if predicted end coordinate is within ground truth end rectangle
                pred_end_x, pred_end_y = float(pred_end[0]), float(pred_end[1])
                end_match = (
                    gt_rect_end["left"] <= pred_end_x <= gt_rect_end["right"]
                    and gt_rect_end["top"] <= pred_end_y <= gt_rect_end["bottom"]
                )
            else:
                # Fall back to tolerance-based comparison if no rectangle provided
                tolerance = 25.0  # Default tolerance
                end_match = (
                    abs(float(pred_end[0]) - float(gt_end[0])) <= tolerance
                    and abs(float(pred_end[1]) - float(gt_end[1])) <= tolerance
                )

            coordinates_match = start_match and end_match

            # Compare other arguments using normalized values
            # Exclude coordinates from comparison as they are handled separately
            other_args_match = True
            for key in ["button", "duration", "key_hold"]:
                pred_val = pred_normalized.get(key)
                gt_val = gt_normalized.get(key)

                # Convert to strings for comparison, handling None values
                pred_str = str(pred_val).lower() if pred_val is not None else "none"
                gt_str = str(gt_val).lower() if gt_val is not None else "none"

                if pred_str != gt_str:
                    other_args_match = False
                    break

            return coordinates_match and other_args_match

        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning(f"Error comparing drag arguments: {e}")
            return False

    def _compare_a11y_args(
        self,
        pred_args: Dict,
        gt_args: Dict,
        gt_rect: Dict = None,
        pred_function: str = None,
        gt_function: str = None,
    ) -> bool:
        """Compare A11Y operation arguments based on model output type.
        
        Prioritizes comparison based on what the model outputs:
        - If model outputs control_label, use gt control_label for comparison
        - If model outputs coordinate, use gt coordinate/rectangle for comparison
        - Both ground truth rect and control_label are available for comparison
        """
        try:
            # If we don't have function names, try to infer from the presence of argument types
            if pred_function is None:
                if "control_label" in pred_args or "coordinate" in pred_args:
                    pred_function = "click"  # Default assumption
                else:
                    pred_function = "unknown"

            if gt_function is None:
                if "control_label" in gt_args or "coordinate" in gt_args:
                    gt_function = "click"  # Default assumption
                else:
                    gt_function = "unknown"

            # Normalize both argument sets with A11Y default values
            pred_normalized = normalize_tool_args_a11y(pred_function, pred_args)
            gt_normalized = normalize_tool_args_a11y(gt_function, gt_args)

            # Priority based on model output: check what the model actually predicted
            model_has_control_label = "control_label" in pred_normalized and pred_normalized["control_label"] is not None
            model_has_coordinate = "coordinate" in pred_normalized and pred_normalized["coordinate"] is not None
            
            # Case 1: Model outputs control_label - prioritize control_label comparison
            if model_has_control_label:
                if "control_label" in gt_normalized and gt_normalized["control_label"] is not None:
                    pred_label = pred_normalized["control_label"]
                    gt_label = gt_normalized["control_label"]
                    
                    control_label_match = str(pred_label) == str(gt_label)
                    
                    # If control labels match, check other non-coordinate arguments
                    if control_label_match:
                        other_args_match = True
                        for key in pred_normalized:
                            if key not in ["control_label", "coordinate"]:
                                pred_val = pred_normalized[key]
                                gt_val = gt_normalized.get(key)

                                # Convert to strings for comparison, handling None values
                                pred_str = (
                                    str(pred_val).lower()
                                    if pred_val is not None
                                    else "none"
                                )
                                gt_str = (
                                    str(gt_val).lower() if gt_val is not None else "none"
                                )

                                if pred_str != gt_str:
                                    other_args_match = False
                                    break

                        return control_label_match and other_args_match
                    else:
                        return False
                else:
                    # Model predicted control_label but GT doesn't have it - fallback to coordinate if available
                    if model_has_coordinate and ("coordinate" in gt_normalized or gt_rect):
                        return self._compare_coordinate_args(pred_normalized, gt_normalized, gt_rect)
                    else:
                        return False
            
            # Case 2: Model outputs coordinate - prioritize coordinate comparison  
            elif model_has_coordinate:
                return self._compare_coordinate_args(pred_normalized, gt_normalized, gt_rect)
            
            # Case 3: Standard comparison for non-coordinate/non-control_label operations
            else:
                return self._compare_standard_args(pred_normalized, gt_normalized)

        except Exception as e:
            self.logger.warning(f"Error comparing A11Y arguments: {e}")
            return False
    
    def _compare_coordinate_args(self, pred_normalized: Dict, gt_normalized: Dict, gt_rect: Dict = None) -> bool:
        """Helper method to compare coordinate-based arguments."""
        if "coordinate" not in pred_normalized:
            return False
            
        pred_coord = pred_normalized["coordinate"]
        
        # Ensure predicted coordinate is valid
        if not (isinstance(pred_coord, (list, tuple)) and len(pred_coord) == 2):
            return False

        coordinate_match = False
        
        # Try rectangle-based comparison first if available
        if gt_rect:
            pred_x, pred_y = float(pred_coord[0]), float(pred_coord[1])
            coordinate_match = (
                gt_rect["left"] <= pred_x <= gt_rect["right"]
                and gt_rect["top"] <= pred_y <= gt_rect["bottom"]
            )
        # Fallback to coordinate-to-coordinate comparison if GT has coordinate
        elif "coordinate" in gt_normalized and gt_normalized["coordinate"] is not None:
            gt_coord = gt_normalized["coordinate"]
            if isinstance(gt_coord, (list, tuple)) and len(gt_coord) == 2:
                tolerance = 25.0  # Default tolerance
                coordinate_match = (
                    abs(float(pred_coord[0]) - float(gt_coord[0])) <= tolerance
                    and abs(float(pred_coord[1]) - float(gt_coord[1])) <= tolerance
                )
        
        if not coordinate_match:
            return False
            
        # Compare other arguments using normalized values (excluding coordinate and control_label)
        other_args_match = True
        for key in pred_normalized:
            if key not in ["coordinate", "control_label"]:
                pred_val = pred_normalized[key]
                gt_val = gt_normalized.get(key)

                # Convert to strings for comparison, handling None values
                pred_str = str(pred_val).lower() if pred_val is not None else "none"
                gt_str = str(gt_val).lower() if gt_val is not None else "none"

                if pred_str != gt_str:
                    other_args_match = False
                    break

        return coordinate_match and other_args_match
    
    def _compare_standard_args(self, pred_normalized: Dict, gt_normalized: Dict) -> bool:
        """Helper method for standard argument comparison."""
        pred_normalized_str = {}
        gt_normalized_str = {}

        for k, v in pred_normalized.items():
            if isinstance(v, (str, bool)):
                pred_normalized_str[k] = str(v).lower()
            elif isinstance(v, (list, tuple)):
                pred_normalized_str[k] = v
            elif v is None:
                pred_normalized_str[k] = "none"
            else:
                pred_normalized_str[k] = v

        for k, v in gt_normalized.items():
            if isinstance(v, (str, bool)):
                gt_normalized_str[k] = str(v).lower()
            elif isinstance(v, (list, tuple)):
                gt_normalized_str[k] = v
            elif v is None:
                gt_normalized_str[k] = "none"
            else:
                gt_normalized_str[k] = v

        return pred_normalized_str == gt_normalized_str

    def transform_a11y(self, a11y):
        """
        Transform a11y information to a more readable format.
        """
        if "application_windows_info" in a11y:
            application_rect = a11y["application_windows_info"]["control_rect"]
        else:
            application_rect = [0, 0, 0, 0]

        for control_info in a11y["uia_controls_info"]:
            control_info["control_rect"] = [
                control_info["control_rect"][0] - application_rect[0],
                control_info["control_rect"][1] - application_rect[1],
                control_info["control_rect"][2] - application_rect[0],
                control_info["control_rect"][3] - application_rect[1]
            ]

        return a11y

    def evaluate_one(
        self, sample: Dict[str, Any], retry: int = 5
    ) -> ActionPredictionA11yEvaluationResult:
        """Evaluate a single sample with A11Y support."""
        start_time = time.time()

        # Create model instance for this thread
        model = self._create_model()

        try:
            # Use annotated image for A11Y evaluation
            annotated_img_path = sample["screenshot_annotated"]
            request = sample["request"]
            action = sample["action"]
            domain = sample["domain"]
            sample_id = sample["sample_id"]
            previous_actions = sample.get("previous_actions", [])
            control_infos = sample.get("control_infos", {})

            # control_infos = self.transform_a11y(control_infos)

            if "qwen" in model.model_name.lower():
                from qwen_vl_utils import smart_resize

                w, h = Image.open(annotated_img_path).size
                resized_height, resized_width = smart_resize(h, w)

                scale_w = resized_width / w
                scale_h = resized_height / h

                # for control in control_infos.get("uia_controls_info"):
                #     control["control_rect"] = [
                #         control["control_rect"][0] * scale_w,
                #         control["control_rect"][1] * scale_h,
                #         control["control_rect"][2] * scale_w,
                #         control["control_rect"][3] * scale_h,
                #     ]

            resolution = Image.open(annotated_img_path).size

            # Extract ground truth action information
            ground_truth_rect = action.get("rectangle", {})

            # Normalize the action arguments
            if action["function"] == "drag":
                # Store original coordinates before deleting
                start_x = action["args"]["start_x"]
                start_y = action["args"]["start_y"]
                end_x = action["args"]["end_x"]
                end_y = action["args"]["end_y"]

                action["args"]["start_coordinate"] = [start_x, start_y]
                action["args"]["end_coordinate"] = [end_x, end_y]

                del action["args"]["start_x"]
                del action["args"]["start_y"]
                del action["args"]["end_x"]
                del action["args"]["end_y"]

                ground_truth_rect = {
                    "left": max(0, start_x) - 25,
                    "top": max(0, start_y) - 25,
                    "right": min(start_x + 25, resolution[0]),
                    "bottom": min(start_y + 25, resolution[1]),
                }

                ground_truth_rect_drag_end = {
                    "left": max(0, end_x) - 25,
                    "top": max(0, end_y) - 25,
                    "right": min(end_x + 25, resolution[0]),
                    "bottom": min(end_y + 25, resolution[1]),
                }

            else:
                if "x" in action["args"]:
                    del action["args"]["x"]
                if "y" in action["args"]:
                    del action["args"]["y"]

                if "coordinate_x" in action and action["coordinate_x"]:
                    action["args"]["coordinate"] = [
                        action["coordinate_x"],
                        action["coordinate_y"],
                    ]

            gt_function = action.get("function", "")
            gt_args = action.get("args", {})
            gt_status = sample.get("status", "")
            
            # Add control_label to gt_args if available
            control_label = action.get("control_label", {})
            if control_label:
                gt_args["control_label"] = control_label

            # Model prediction
            retry_count = 0
            raw_response = ""
            thoughts = ""
            pred_function = None
            pred_args = None
            pred_status = None

            while retry_count < retry:
                try:
                    # Select A11Y actions based on domain
                    actions = ""
                    if domain.lower() == "word":
                        actions = SUPPORTED_ACTIONS_WORD_A11Y
                    elif domain.lower() == "excel":
                        actions = SUPPORTED_ACTIONS_EXCEL_A11Y
                    elif domain.lower() == "ppt":
                        actions = SUPPORTED_ACTIONS_PPT_A11Y

                    # Construct A11Y prompts using model's method
                    system_prompt, user_prompt = model.construct_action_a11y_prompt(
                        instruction=request,
                        history=previous_actions,
                        actions=actions,
                        control_infos=control_infos,
                    )

                    # Use universal predict method with annotated image
                    raw_response = model.predict(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_path=annotated_img_path,
                        temperature=0.0,
                        max_tokens=4096,
                    )

                    # Parse thoughts
                    thoughts = model.parse_thoughts(raw_response)

                    # Parse tool call information using model's parse_action method
                    pred_function, pred_args, pred_status = model.parse_action(
                        raw_response
                    )

                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.warning(
                        f"Prediction failed for sample {sample_id}, retry {retry_count}/{retry}: {e.with_traceback()}"
                    )
                    time.sleep(5)

            if retry_count >= retry:
                self.logger.warning(
                    f"Max retries reached for sample {sample_id}, using default values"
                )
                pred_function = None
                pred_args = {}
                pred_status = None
                raw_response = "ERROR: Max retries reached"
                thoughts = (
                    "ERROR: Unable to generate thoughts due to prediction failures"
                )

            # Compare predictions with ground truth
            # For drag operations, pass the rectangle information
            if gt_function == "drag":
                function_match, args_match, status_match = self.compare_actions(
                    pred_function,
                    pred_args,
                    pred_status,
                    gt_function,
                    gt_args,
                    gt_status,
                    ground_truth_rect,
                    locals().get("ground_truth_rect_drag_end"),
                )
            else:
                function_match, args_match, status_match = self.compare_actions(
                    pred_function,
                    pred_args,
                    pred_status,
                    gt_function,
                    gt_args,
                    gt_status,
                    ground_truth_rect,
                )

            # Overall success: all three components must match
            success = function_match and args_match and status_match

            execution_time = time.time() - start_time

            # Create detailed output for different operation types with A11Y info
            if gt_function == "drag":
                pred_start = (
                    pred_args.get("start_coordinate", "N/A") if pred_args else "N/A"
                )
                pred_end = (
                    pred_args.get("end_coordinate", "N/A") if pred_args else "N/A"
                )
                gt_start = gt_args.get("start_coordinate", "N/A")
                gt_end = gt_args.get("end_coordinate", "N/A")

                print(
                    f"Sample {sample_id} [A11Y-DRAG]: "
                    f"Function: {pred_function} vs {gt_function} ({'✓' if function_match else '✗'}), "
                    f"Start: {pred_start} vs {gt_start}, "
                    f"End: {pred_end} vs {gt_end}, "
                    f"Args: ({'✓' if args_match else '✗'}), "
                    f"Status: {pred_status} vs {gt_status} ({'✓' if status_match else '✗'}), "
                    f"Overall: {'✓' if success else '✗'}"
                )
            elif pred_args and ("control_label" in pred_args or "coordinate" in pred_args):
                pred_label = pred_args.get("control_label", "N/A")
                pred_coord = pred_args.get("coordinate", "N/A")
                gt_label = gt_args.get("control_label", "N/A")
                gt_coord = gt_args.get("coordinate", "N/A")

                print(
                    f"Sample {sample_id} [A11Y-CTRL]: "
                    f"Function: {pred_function} vs {gt_function} ({'✓' if function_match else '✗'}), "
                    f"Label: {pred_label} vs {gt_label}, "
                    f"Coord: {pred_coord} vs {gt_coord}, "
                    f"Args: ({'✓' if args_match else '✗'}), "
                    f"Status: {pred_status} vs {gt_status} ({'✓' if status_match else '✗'}), "
                    f"Overall: {'✓' if success else '✗'}"
                )
            else:
                print(
                    f"Sample {sample_id} [A11Y]: "
                    f"Function: {pred_function} vs {gt_function} ({'✓' if function_match else '✗'}), "
                    f"Args: {pred_args} vs {gt_args} ({'✓' if args_match else '✗'}), "
                    f"Status: {pred_status} vs {gt_status} ({'✓' if status_match else '✗'}), "
                    f"Overall: {'✓' if success else '✗'}"
                )

            return ActionPredictionA11yEvaluationResult(
                success=success,
                predicted_function=pred_function,
                predicted_args=pred_args,
                predicted_status=pred_status,
                ground_truth_function=gt_function,
                ground_truth_args=gt_args,
                ground_truth_status=gt_status,
                function_match=function_match,
                args_match=args_match,
                status_match=status_match,
                thoughts=thoughts,
                ground_truth_rect=ground_truth_rect,
                sample_id=sample_id,
                execution_time=execution_time,
                raw_model_output=raw_response,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Log entire error with traceback
            self.logger.error(
                f"Error evaluating sample {sample.get('sample_id', 'unknown')}: {traceback.format_exc()}"
            )

            return ActionPredictionA11yEvaluationResult(
                success=False,
                predicted_function=None,
                predicted_args={},
                predicted_status=None,
                ground_truth_function=sample.get("action", {}).get("function", ""),
                ground_truth_args=sample.get("action", {}).get("args", {}),
                ground_truth_status=sample.get("action", {}).get("status", ""),
                function_match=False,
                args_match=False,
                status_match=False,
                ground_truth_rect=sample.get("action", {}).get("rectangle", {}),
                sample_id=sample.get("sample_id", "unknown"),
                execution_time=execution_time,
                error_message=str(e),
                raw_model_output="ERROR: Exception occurred during evaluation",
            )
