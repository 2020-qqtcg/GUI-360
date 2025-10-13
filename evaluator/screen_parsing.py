import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import json5
from PIL import Image

from config import EvaluationConfig
from evaluator.base import BaseEvaluator, EvaluationResult
from models.mock_model import MockModel

class TextSimilarityCalculator:
    """A class to calculate text similarity using Sentence-BERT."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the TextSimilarityCalculator with a Sentence-BERT model.

        Args:
            model_name (str): The name of the pre-trained model to use.
                              'all-MiniLM-L6-v2' is a good, fast choice.
        """
        try:
            self.device = 'cpu'
            print(f"Using device: {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings using Sentence-BERT.

        Args:
            text1 (str): The first string.
            text2 (str): The second string.

        Returns:
            float: A similarity score between 0 and 1.
                   Returns -1.0 if the model is not loaded or an error occurs.
        """
        if not self.model:
            return -1.0

        try:
            text1 = text1.strip() if text1 else ""
            text2 = text2.strip() if text2 else ""

            if not text1 or not text2:
                return 0.0

            embeddings = self.model.encode([text1, text2], convert_to_tensor=True, device=self.device)

            cosine_scores = util.cos_sim(embeddings[0], embeddings[1])

            similarity_score = cosine_scores.item()
            return similarity_score

        except Exception as e:
            print(f"An error occurred during similarity calculation: {e}")
            return -1.0

@dataclass
class ScreenParsingEvaluationResult(EvaluationResult):
    """Result of a single screen parsing evaluation."""

    predicted_control_infos: Optional[List[Dict[str, Any]]] = None
    ground_truth_control_infos: Optional[List[Dict[str, Any]]] = None
    recall: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    control_text_similarity: float = 0.0
    iou_accuracy: float = 0.0
    total_text_success: float = 0.0
    pred_count: float = 0.0
    thoughts: Optional[str] = None


class ScreenParsingEvaluator(BaseEvaluator):
    """Evaluator for screen parsing tasks."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self._lock = threading.Lock()
        self.results: List[ScreenParsingEvaluationResult] = []
        self.sentence_bert = TextSimilarityCalculator()

    def load_data(self, task_type):
        """Load data from dataset directory."""
        root_path = os.path.join(self.config.root_dir)
        sample_count = 0

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
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line_num, line in enumerate(f, 1):
                                if not line.strip():
                                    continue

                                try:
                                    data = json.loads(line.strip())

                                    # Create sample ID
                                    sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{line_num}"

                                    # Build image paths - use clean image for screen parsing
                                    clean_img_path = os.path.join(
                                        root_path,
                                        "image",
                                        domain,
                                        category,
                                        data["step"]["screenshot_clean"],
                                    )

                                    # Validate image exists
                                    if not os.path.exists(clean_img_path):
                                        self.logger.warning(
                                            f"Image not found: {clean_img_path}"
                                        )
                                        continue

                                    # Check if this sample has control_infos for screen parsing
                                    if "control_infos" in data["step"]:
                                        control_infos = data["step"]["control_infos"]

                                        # Only process samples with valid control_infos
                                        if control_infos and isinstance(
                                                control_infos, dict
                                        ):
                                            sample = {
                                                "sample_id": sample_id,
                                                "request": data["request"],
                                                "screenshot_clean": clean_img_path,
                                                "domain": domain,
                                                "category": category,
                                                "control_infos": control_infos,
                                            }

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
                                    self.logger.error(
                                        f"Error in {file_path}:{line_num} - {e}"
                                    )
                                    continue

                    except Exception as e:
                        self.logger.error(f"Error reading file {file_path}: {e}")
                        continue

        self.logger.info(f"Loaded {sample_count} samples total")

    def load_previous_results(
            self, results_file_path: str
    ) -> Tuple[List[ScreenParsingEvaluationResult], List[str]]:
        """Load previous evaluation results and extract error cases."""
        try:
            with open(results_file_path, "r", encoding="utf-8") as f:
                data = json5.load(f)

            previous_results = []
            error_sample_ids = []

            if "detailed_results" in data:
                for result_data in data["detailed_results"]:
                    # Reconstruct ScreenParsingEvaluationResult objects
                    result = ScreenParsingEvaluationResult(
                        success=result_data.get("success", False),
                        predicted_control_infos=result_data.get(
                            "predicted_control_infos"
                        ),
                        ground_truth_control_infos=result_data.get(
                            "ground_truth_control_infos"
                        ),
                        recall=result_data.get(
                            "recall", result_data.get("control_match_accuracy", 0.0)
                        ),  # 向后兼容
                        precision=result_data.get("precision", 0.0),
                        control_text_similarity=result_data.get(
                            "control_text_similarity",
                            result_data.get("control_text_accuracy", 0.0),
                        ),  # 向后兼容
                        iou_accuracy=result_data.get("iou_accuracy", 0.0),
                        pred_count=result_data.get("pred_count", 0.0),
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

    def create_error_result(
            self,
            sample: Dict[str, Any],
            error_message: str,
            execution_time: float = 0.0,
            raw_output: str = "ERROR",
    ) -> ScreenParsingEvaluationResult:
        """Create an error result object for screen parsing."""
        return ScreenParsingEvaluationResult(
            success=False,
            predicted_control_infos=[],
            ground_truth_control_infos=sample.get("control_infos", {}).get(
                "uia_controls_info", []
            ),
            recall=0.0,
            precision=0.0,
            control_text_similarity=0.0,
            iou_accuracy=0.0,
            pred_count=0.0,
            ground_truth_rect={},
            sample_id=sample.get("sample_id", "unknown"),
            execution_time=execution_time,
            error_message=error_message,
            raw_model_output=raw_output,
        )

    def calculate_iou(self, rect1: Dict, rect2: Dict) -> float:
        """Calculate IoU (Intersection over Union) between two rectangles."""
        try:
            # Extract coordinates
            x1_min, y1_min, x1_max, y1_max = (
                rect1["left"],
                rect1["top"],
                rect1["right"],
                rect1["bottom"],
            )
            x2_min, y2_min, x2_max, y2_max = (
                rect2["left"],
                rect2["top"],
                rect2["right"],
                rect2["bottom"],
            )

            # Calculate intersection
            x_min = max(x1_min, x2_min)
            y_min = max(y1_min, y2_min)
            x_max = min(x1_max, x2_max)
            y_max = min(y1_max, y2_max)

            if x_max <= x_min or y_max <= y_min:
                return 0.0

            intersection = (x_max - x_min) * (y_max - y_min)

            # Calculate union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection

            if union <= 0:
                return 0.0

            return intersection / union

        except (KeyError, TypeError, ValueError):
            return 0.0

    def match_control_infos(
            self,
            pred_controls: List[Dict],
            gt_controls: List[Dict],
            iou_threshold: float = 0.3,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Match predicted control infos with ground truth and calculate accuracies.

        Returns:
            Tuple of (recall, precision, control_text_similarity, iou_accuracy, total_text_success, f1_score, pred_count)
        """
        if not pred_controls or not gt_controls:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Find best matches based on IoU threshold
        matched_controls = []
        total_iou = 0.0

        for gt_control in gt_controls:
            best_match = None
            best_iou = 0.0

            for pred_control in pred_controls:
                # Skip if pred_control is not a dictionary
                if not isinstance(pred_control, dict):
                    continue

                # Skip if already matched
                if any(pred_control is match[0] for match in matched_controls):
                    continue

                # Calculate IoU based on control_rect
                if "control_rect" in pred_control and "control_rect" in gt_control:
                    # Convert list format [left, top, right, bottom] to dict format
                    pred_rect = pred_control["control_rect"]
                    gt_rect = gt_control["control_rect"]

                    if isinstance(pred_rect, list) and len(pred_rect) >= 4:
                        pred_rect = {
                            "left": pred_rect[0],
                            "top": pred_rect[1],
                            "right": pred_rect[2],
                            "bottom": pred_rect[3],
                        }
                    elif isinstance(pred_rect, list) and len(pred_rect) == 2:
                        # Assume it's [width, height] from top-left
                        pred_rect = {
                            "left": 0,
                            "top": 0,
                            "right": pred_rect[0],
                            "bottom": pred_rect[1],
                        }

                    if isinstance(gt_rect, list) and len(gt_rect) >= 4:
                        gt_rect = {
                            "left": gt_rect[0],
                            "top": gt_rect[1],
                            "right": gt_rect[2],
                            "bottom": gt_rect[3],
                        }
                    elif isinstance(gt_rect, list) and len(gt_rect) == 2:
                        # Assume it's [width, height] from top-left
                        gt_rect = {
                            "left": 0,
                            "top": 0,
                            "right": gt_rect[0],
                            "bottom": gt_rect[1],
                        }

                    iou = self.calculate_iou(pred_rect, gt_rect)

                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match = pred_control

            if best_match:
                matched_controls.append((best_match, gt_control, best_iou))
                total_iou += best_iou

        # Calculate accuracies
        num_matched = len(matched_controls)
        num_gt = len(gt_controls)
        num_pred = len(pred_controls)

        # Recall: ratio of matched controls to ground truth controls
        # 有多少个真实控件被成功找到了
        recall = num_matched / num_gt if num_gt > 0 else 0.0

        # Precision: ratio of matched controls to predicted controls
        # 有多少个预测控件是正确的
        precision = num_matched / num_pred if num_pred > 0 else 0.0

        f1_score = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )

        # Average IoU for matched controls
        iou_accuracy = total_iou / num_matched if num_matched > 0 else 0.0

        # Control text similarity: average similarity of control texts among matched controls
        total_text_similarity = 0.0
        total_text_success = 0
        for pred_control, gt_control, _ in matched_controls:
            pred_text = pred_control.get("control_text", "")
            gt_text = gt_control.get("control_text", "")
            similarity = self.sentence_bert.calculate_text_similarity(pred_text, gt_text)

            if similarity > 0.5:
                total_text_success += 1
            total_text_similarity += similarity

        control_text_similarity = (
            total_text_similarity / num_matched if num_matched > 0 else 0.0
        )

        pred_count = len(pred_controls)

        return (
            recall,
            precision,
            f1_score,
            control_text_similarity,
            iou_accuracy,
            total_text_success,
            pred_count
        )

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate detailed evaluation statistics."""
        if not self.results:
            return {}

        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        error_results = [r for r in self.results if r.error_message]

        total_samples = len(self.results)
        success_count = len(successful_results)

        # Basic metrics
        success_rate = success_count / total_samples * 100 if total_samples > 0 else 0
        error_rate = (
            len(error_results) / total_samples * 100 if total_samples > 0 else 0
        )

        # Timing statistics
        execution_times = [r.execution_time for r in self.results]
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )

        # Calculate average metrics across all samples
        recalls = [r.recall for r in self.results]
        precisions = [r.precision for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        text_similarities = [r.control_text_similarity for r in self.results]
        iou_accuracies = [r.iou_accuracy for r in self.results]
        pred_counts = [r.pred_count for r in self.results]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_text_similarity = sum(text_similarities) / len(text_similarities) if text_similarities else 0.0
        avg_iou_accuracy = sum(iou_accuracies) / len(iou_accuracies) if iou_accuracies else 0.0
        avg_pred_count = sum(pred_counts) / len(pred_counts) if pred_counts else 0.0

        # Domain/category breakdown
        domain_stats = {}
        for result in self.results:
            sample_parts = result.sample_id.split("_")
            if len(sample_parts) >= 2:
                domain = sample_parts[0]
                category = sample_parts[1]

                if domain not in domain_stats:
                    domain_stats[domain] = {
                        "total": 0,
                        "success": 0,
                        "categories": {},
                        "recalls": [],
                        "precisions": [],
                        "f1_scores": [],
                        "text_similarities": [],
                        "iou_accuracies": [],
                        "pred_counts": []
                    }

                domain_stats[domain]["total"] += 1
                domain_stats[domain]["recalls"].append(result.recall)
                domain_stats[domain]["precisions"].append(result.precision)
                domain_stats[domain]["f1_scores"].append(result.f1_score)
                domain_stats[domain]["text_similarities"].append(result.control_text_similarity)
                domain_stats[domain]["iou_accuracies"].append(result.iou_accuracy)
                domain_stats[domain]["pred_counts"].append(result.pred_count)
                if result.success:
                    domain_stats[domain]["success"] += 1

                if category not in domain_stats[domain]["categories"]:
                    domain_stats[domain]["categories"][category] = {
                        "total": 0,
                        "success": 0,
                        "recalls": [],
                        "precisions": [],
                        "f1_scores": [],
                        "text_similarities": [],
                        "iou_accuracies": [],
                        "pred_counts": []
                    }

                cat_data = domain_stats[domain]["categories"][category]
                cat_data["total"] += 1
                cat_data["recalls"].append(result.recall)
                cat_data["precisions"].append(result.precision)
                cat_data["f1_scores"].append(result.f1_score)
                cat_data["text_similarities"].append(result.control_text_similarity)
                cat_data["iou_accuracies"].append(result.iou_accuracy)
                cat_data["pred_counts"].append(result.pred_count)
                if result.success:
                    cat_data["success"] += 1

        # Calculate success rates and average metrics for domains/categories
        for domain in domain_stats:
            domain_data = domain_stats[domain]
            if domain_data["total"] > 0:
                domain_data["success_rate"] = (
                        domain_data["success"] / domain_data["total"] * 100
                )
                # Calculate domain averages
                domain_data["avg_recall"] = sum(domain_data["recalls"]) / len(domain_data["recalls"])
                domain_data["avg_precision"] = sum(domain_data["precisions"]) / len(domain_data["precisions"])
                domain_data["avg_f1_score"] = sum(domain_data["f1_scores"]) / len(domain_data["f1_scores"])
                domain_data["avg_text_similarity"] = sum(domain_data["text_similarities"]) / len(
                    domain_data["text_similarities"])
                domain_data["avg_iou_accuracy"] = sum(domain_data["iou_accuracies"]) / len(
                    domain_data["iou_accuracies"])

            for category in domain_data["categories"]:
                cat_data = domain_data["categories"][category]
                if cat_data["total"] > 0:
                    cat_data["success_rate"] = (
                            cat_data["success"] / cat_data["total"] * 100
                    )
                    # Calculate category averages
                    cat_data["avg_recall"] = sum(cat_data["recalls"]) / len(cat_data["recalls"])
                    cat_data["avg_precision"] = sum(cat_data["precisions"]) / len(cat_data["precisions"])
                    cat_data["avg_f1_score"] = sum(cat_data["f1_scores"]) / len(cat_data["f1_scores"])
                    cat_data["avg_text_similarity"] = sum(cat_data["text_similarities"]) / len(
                        cat_data["text_similarities"])
                    cat_data["avg_iou_accuracy"] = sum(cat_data["iou_accuracies"]) / len(cat_data["iou_accuracies"])
                    cat_data["avg_pred_count"] = sum(cat_data["pred_counts"]) / len(cat_data["pred_counts"])
                    # Clean up temporary lists to save memory
                    del cat_data["recalls"], cat_data["precisions"], cat_data["f1_scores"]
                    del cat_data["text_similarities"], cat_data["iou_accuracies"]
                    del cat_data["pred_counts"]
            # Clean up temporary lists for domain
            del domain_data["recalls"], domain_data["precisions"], domain_data["f1_scores"]
            del domain_data["text_similarities"], domain_data["iou_accuracies"]
            del domain_data["pred_counts"]
        return {
            "total_samples": total_samples,
            "success_count": success_count,
            "success_rate": success_rate,
            "error_count": len(error_results),
            "error_rate": error_rate,
            "avg_execution_time": avg_execution_time,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "avg_f1_score": avg_f1_score,
            "avg_pred_count": avg_pred_count,
            "avg_text_similarity": avg_text_similarity,
            "avg_iou_accuracy": avg_iou_accuracy,
            "domain_stats": domain_stats,
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def transfer_control_infos(self, control_infos: List[Dict[str, Any]], application_rect: List[float]) -> List[
        Dict[str, Any]]:
        for control_info in control_infos:
            control_info["control_rect"] = [
                control_info["control_rect"][0] - application_rect[0],
                control_info["control_rect"][1] - application_rect[1],
                control_info["control_rect"][2] - application_rect[0],
                control_info["control_rect"][3] - application_rect[1],
            ]
        return control_infos

    def evaluate_one(
            self, sample: Dict[str, Any], retry: int = 5
    ) -> ScreenParsingEvaluationResult:
        """Evaluate a single sample for screen parsing."""
        start_time = time.time()

        # Create model instance for this thread
        model = self._create_model()

        try:
            # Extract sample information
            clean_img_path = sample["screenshot_clean"]
            control_infos = sample["control_infos"]
            sample_id = sample["sample_id"]
            domain = sample["domain"]

            # Extract ground truth control infos
            gt_uia_controls = control_infos.get("uia_controls_info", [])
            application_infos = control_infos.get("application_windows_info", None)
            if application_infos is not None:
                application_rect = application_infos.get("control_rect", [0, 0, 0, 0])
            else:
                application_rect = [0, 0, 0, 0]

            gt_uia_controls = self.transfer_control_infos(gt_uia_controls, application_rect)
            # Model prediction
            retry_count = 0
            raw_response = ""
            thoughts = ""
            pred_control_infos = []

            resolution = Image.open(clean_img_path).size

            while retry_count < retry:
                try:
                    # Construct screen parsing prompt using model's method
                    system_prompt, user_prompt = (
                        model.construct_screen_parsing_prompt(
                            resolution=resolution
                        )
                    )

                    # Use universal predict method with clean image
                    if isinstance(model, MockModel):
                        raw_response = model.predict(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            image_path=clean_img_path,
                            temperature=0.0,
                            max_tokens=16384,
                            sample_id=sample_id,
                        )
                    else:
                        raw_response = model.predict(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            image_path=clean_img_path,
                            temperature=0.0,
                            max_tokens=16384,
                        )

                    # Parse thoughts if available
                    if hasattr(model, "parse_thoughts"):
                        thoughts = raw_response
                    else:
                        thoughts = "Thoughts not available for this model"

                    # Parse control infos from model response
                    if hasattr(model, "parse_screen_parsing"):
                        pred_control_infos = model.parse_screen_parsing(
                            raw_response
                        )
                    else:
                        # Fallback parsing logic
                        pred_control_infos = self._parse_control_infos_fallback(
                            raw_response
                        )

                    # Ensure pred_control_infos is a list of dictionaries
                    if not isinstance(pred_control_infos, list):
                        self.logger.warning(
                            f"Expected list but got {type(pred_control_infos)}, converting to list"
                        )
                        pred_control_infos = (
                            [pred_control_infos]
                            if isinstance(pred_control_infos, dict)
                            else []
                        )

                    # Filter out non-dictionary items
                    pred_control_infos = [
                        item for item in pred_control_infos if isinstance(item, dict)
                    ]

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
                pred_control_infos = []
                raw_response = "ERROR: Max retries reached"
                thoughts = (
                    "ERROR: Unable to generate thoughts due to prediction failures"
                )

            # Calculate evaluation metrics
            match_threshold = 0.5  # IoU threshold for matching

            recall, precision, f1, text_similarity, iou_acc, total_text_success, pred_count = (
                self.match_control_infos(
                    pred_control_infos, gt_uia_controls, iou_threshold=match_threshold
                )
            )

            # Overall success: if recall (formerly control_match_accuracy) is above threshold
            success = recall >= 0.1

            execution_time = time.time() - start_time

            # Print evaluation results
            print(
                f"Sample {sample_id} [SCREEN-PARSING]: "
                f"Controls: {len(pred_control_infos)} vs {len(gt_uia_controls)}, "
                f"Recall: {recall:.3f}, "
                f"Precision: {precision:.3f}, "
                f"F1 score: {f1:.3f}, "
                f"TextSim: {text_similarity:.3f}, "
                f"IoU: {iou_acc:.3f}, "
                f"TextSuccess: {total_text_success:.3f}, "
                f"PredCount: {pred_count}, "
                f"success: {success}, "
            )

            return ScreenParsingEvaluationResult(
                success=success,
                predicted_control_infos=pred_control_infos,
                ground_truth_control_infos=gt_uia_controls,
                recall=recall,
                precision=precision,
                f1_score=f1,
                control_text_similarity=text_similarity,
                iou_accuracy=iou_acc,
                pred_count=pred_count,
                thoughts=thoughts,
                total_text_success=total_text_success,
                ground_truth_rect={},  # Not applicable for screen parsing
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

            return ScreenParsingEvaluationResult(
                success=False,
                predicted_control_infos=[],
                ground_truth_control_infos=sample.get("control_infos", {}).get(
                    "uia_controls_info", []
                ),
                recall=0.0,
                precision=0.0,
                control_text_similarity=0.0,
                iou_accuracy=0.0,
                ground_truth_rect={},
                sample_id=sample.get("sample_id", "unknown"),
                execution_time=execution_time,
                error_message=str(e),
                raw_model_output="ERROR: Exception occurred during evaluation",
            )

    def _parse_control_infos_fallback(self, raw_response: str) -> List[Dict[str, Any]]:
        """
        Fallback method to parse control infos from raw model response.
        This method tries to extract JSON-like structures from the response.
        """
        try:
            # Try to find JSON in the response
            import re

            # Look for JSON arrays or objects that might contain control info
            json_pattern = r"\[.*?\]|\{.*?\}"
            matches = re.findall(json_pattern, raw_response, re.DOTALL)

            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Check if it looks like control infos
                        if isinstance(parsed[0], dict) and any(
                                key in parsed[0] for key in ["control_text", "control_rect"]
                        ):
                            return parsed
                    elif isinstance(parsed, dict):
                        # Single control info
                        if any(
                                key in parsed for key in ["control_text", "control_rect"]
                        ):
                            return [parsed]
                except json.JSONDecodeError:
                    continue

            # If no valid JSON found, return empty list
            return []

        except Exception as e:
            self.logger.warning(f"Failed to parse control infos from response: {e}")
            return []
