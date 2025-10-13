import json
import os
import threading
import time
import traceback

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


from config import EvaluationConfig
from evaluator.base import BaseEvaluator, EvaluationResult
from prompts.prompt_grounding import GROUNDING_SYS_PROMPT, GROUNDING_USER_PROMPT


@dataclass
class GroundingEvaluationResult(EvaluationResult):
    """Result of a single evaluation."""

    predicted_coords: Tuple[float, float] = (0.0, 0.0)
    thoughts: Optional[str] = None


class GroundingEvaluator(BaseEvaluator):
    """Evaluator for GUI grounding tasks."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self._lock = threading.Lock()
        self.results: List[GroundingEvaluationResult] = []
        
    def create_error_result(self, sample: Dict[str, Any], error_message: str, execution_time: float = 0.0, raw_output: str = "ERROR") -> GroundingEvaluationResult:
        """Create a grounding error result object with predicted_coords."""
        return GroundingEvaluationResult(
            success=False,
            predicted_coords=(0.0, 0.0),
            thoughts="ERROR: Unable to generate thoughts due to error",
            ground_truth_rect=sample.get("action", {}).get("rectangle", {}),
            sample_id=sample.get("sample_id", "unknown"),
            execution_time=execution_time,
            error_message=error_message,
            raw_model_output=raw_output,
        )

    def load_previous_results(
        self, results_file_path: str
    ) -> Tuple[List[GroundingEvaluationResult], List[str]]:
        """Load previous evaluation results and extract error cases.

        Returns:
            Tuple of (all_previous_results, error_sample_ids)
        """
        try:
            with open(results_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            previous_results = []
            error_sample_ids = []

            if "detailed_results" in data:
                for result_data in data["detailed_results"]:
                    # Reconstruct EvaluationResult objects
                    result = GroundingEvaluationResult(
                        success=result_data.get("success", False),
                        predicted_coords=tuple(
                            result_data.get("predicted_coords", (0.0, 0.0))
                        ),
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

    def evaluate_one(
        self, sample: Dict[str, Any], retry: int = 5
    ) -> GroundingEvaluationResult:
        """Evaluate a single sample."""
        start_time = time.time()

        # Create model instance for this thread
        model = self._create_model()

        try:
            clean_img_path = sample["screenshot_clean"]
            thought = sample["thought"]
            action = sample["action"]
            sample_id = sample["sample_id"]
            resolution = Image.open(clean_img_path).size

            ground_truth_rect = action["rectangle"]

            # Model prediction
            retry_count = 0
            raw_response = ""
            thoughts = ""

            while retry_count < retry:
                try:
                    # Construct prompts using model's method
                    system_prompt, user_prompt = model.construct_grounding_prompt(
                        thought=thought,
                        resolution=resolution
                    )

                    # Use universal predict method
                    raw_response = model.predict(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_path=clean_img_path,
                        temperature=0.0
                    )
                    predicted_coords = model.parse_coordinates(raw_response)
                    thoughts = model.parse_thoughts(raw_response)

                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.warning(
                        f"Prediction failed for sample {sample_id}, retry {retry_count}/{retry}: {e.with_traceback()}"
                    )
                    time.sleep(5)
                if retry_count >= retry:
                    self.logger.warning(
                        f"Max retries reached for sample {sample_id}, using default coordinates (0.0, 0.0)"
                    )
                    predicted_coords = (0.0, 0.0)
                    raw_response = "ERROR: Max retries reached"
                    thoughts = (
                        "ERROR: Unable to generate thoughts due to prediction failures"
                    )

            x, y = float(predicted_coords[0]), float(predicted_coords[1])

            # Check if prediction is within ground truth rectangle
            success = (
                ground_truth_rect["left"] <= x <= ground_truth_rect["right"]
                and ground_truth_rect["top"] <= y <= ground_truth_rect["bottom"]
            )

            execution_time = time.time() - start_time

            print(
                f"Thoughts for sample {sample_id}: {thoughts}, predicted coordinates: {predicted_coords}, ground truth rectangle: {ground_truth_rect}, success: {success}"
            )

            return GroundingEvaluationResult(
                success=success,
                predicted_coords=(x, y),
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

            return GroundingEvaluationResult(
                success=False,
                predicted_coords=(0.0, 0.0),
                ground_truth_rect=sample.get("action", {}).get("rectangle", {}),
                sample_id=sample.get("sample_id", "unknown"),
                execution_time=execution_time,
                error_message=str(e),
                raw_model_output="ERROR: Exception occurred during evaluation",
            )
