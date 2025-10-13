import json
import logging
import os
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from tqdm import tqdm

from config import EvaluationConfig
from models.base import BaseModel


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""

    success: bool
    ground_truth_rect: Dict[str, float]
    sample_id: str
    execution_time: float
    error_message: Optional[str] = None
    raw_model_output: Optional[str] = None


class BaseEvaluator(ABC):
    """Base evaluator class with improved functionality."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.results: List[EvaluationResult] = []

        import threading
        self._lock = threading.Lock()

        os.makedirs(config.output_dir, exist_ok=True)

    def _create_model(self):
        """Create a model instance for this evaluator."""
        from evaluation import ModelFactory
        return ModelFactory.create_model(
            self.config.model_type, self.config.model_name, self.config.result_path, self.config.api_url
        )

    def create_error_result(self, sample: Dict[str, Any], error_message: str, execution_time: float = 0.0,
                            raw_output: str = "ERROR") -> EvaluationResult:
        """Create an error result object. Subclasses can override this to create specific result types."""
        return EvaluationResult(
            success=False,
            ground_truth_rect=sample.get("action", {}).get("rectangle", {}),
            sample_id=sample.get("sample_id", "unknown"),
            execution_time=execution_time,
            error_message=error_message,
            raw_model_output=raw_output,
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for evaluation."""
        logger_name = f"{self.__class__.__name__}_{id(self)}"
        logger = logging.getLogger(logger_name)

        if not logger.handlers:
            logger.setLevel(getattr(logging, self.config.log_level.upper()))

            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            logger.propagate = False

        return logger

    def load_data(self, task_type) -> Generator[Dict[str, Any], None, None]:
        """Load data from dataset directory."""
        root_path = Path(self.config.root_dir)
        sample_count = 0

        if not root_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_path}")

        data_path = root_path / "data"
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        self.logger.info(f"Loading data from: {root_path}")

        # Process /data directory
        domain_folders = [
            d for d in os.listdir(data_path) if os.path.isdir(data_path / d)
        ]
        self.logger.info(f"Found {len(domain_folders)} domains: {domain_folders}")

        for domain in domain_folders:
            domain_path = data_path / domain
            category_folders = [
                c for c in os.listdir(domain_path) if os.path.isdir(domain_path / c)
            ]

            for category in category_folders:
                category_path = domain_path / category / "success"
                if not category_path.exists():
                    continue

                jsonl_files = [
                    f for f in os.listdir(category_path) if f.endswith(".jsonl")
                ]
                self.logger.debug(
                    f"Processing {len(jsonl_files)} files in {domain}/{category}"
                )

                for jsonl_file in jsonl_files:
                    file_path = category_path / jsonl_file

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line_num, line in enumerate(f, 1):
                                if not line.strip():
                                    continue

                                try:
                                    data = json.loads(line.strip())

                                    # Create sample ID
                                    sample_id = f"{domain}_{category}_{Path(jsonl_file).stem}_{line_num}"

                                    # Build image paths
                                    clean_img_path = (
                                            root_path
                                            / "image"
                                            / domain
                                            / category
                                            / data["step"]["screenshot_clean"]
                                    )
                                    annotated_img_path = (
                                            root_path
                                            / "image"
                                            / domain
                                            / category
                                            / data["step"]["screenshot_annotated"]
                                    )

                                    # Validate image exists
                                    if not clean_img_path.exists():
                                        self.logger.warning(
                                            f"Image not found: {clean_img_path}"
                                        )
                                        continue

                                    if task_type in data["step"]["tags"]:

                                        # Grounding need rectangle
                                        if task_type == "grounding":
                                            if not (
                                                    rectangle := data["step"]["action"].get(
                                                        "rectangle", {}
                                                    )
                                            ):
                                                self.logger.info(
                                                    f"Rectangle wrong for sample {sample_id}: {rectangle}"
                                                )
                                                continue

                                        yield {
                                            "sample_id": sample_id,
                                            "request": data["request"],
                                            "screenshot_clean": str(clean_img_path),
                                            "screenshot_annotated": str(
                                                annotated_img_path
                                            ),
                                            "thought": data["step"]["thought"],
                                            "action": data["step"]["action"],
                                            "domain": domain,
                                            "category": category,
                                        }

                                        sample_count += 1
                                        if (
                                                self.config.max_samples
                                                and sample_count >= self.config.max_samples
                                        ):
                                            self.logger.info(
                                                f"Reached max samples limit: {self.config.max_samples}"
                                            )
                                            return

                                except json.JSONDecodeError as e:
                                    self.logger.error(
                                        f"JSON decode error in {file_path}:{line_num} - {e}"
                                    )
                                    continue

                    except Exception as e:
                        self.logger.error(f"Error reading file {file_path}: {e}")
                        continue

        self.logger.info(f"Loaded {sample_count} samples total")

    def load_previous_results(
            self, resume_from: str
    ) -> Tuple[List[EvaluationResult], List[str]]:
        """Load previous results from a file for resuming evaluation."""
        return [], []

    def evaluate(
            self, thread_num: int = 5, resume_from: str = None
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Run complete evaluation with multi-threading."""
        previous_results = []
        error_sample_ids = []

        if resume_from:
            self.logger.info(f"Resuming evaluation from: {resume_from}")
            previous_results, error_sample_ids = self.load_previous_results(resume_from)
            self.logger.info(
                f"Starting grounding evaluation with {thread_num} threads (RESUME MODE)..."
            )
        else:
            self.logger.info(
                f"Starting grounding evaluation with {thread_num} threads..."
            )

        # Determine task type from config
        task_type = self.config.eval_type if hasattr(self.config, 'eval_type') else "grounding"
        data_generator = self.load_data(task_type=task_type)
        all_samples = list(data_generator)  # Convert to list for progress bar

        if not all_samples:
            raise ValueError("No valid samples found in dataset")

        # Filter samples if resuming
        if resume_from and error_sample_ids:
            samples_to_process = self.filter_samples_for_retry(
                all_samples, error_sample_ids
            )
            if not samples_to_process:
                self.logger.warning(
                    "No error samples found to retry. All samples were successful in previous run."
                )
                # Return previous results as final results
                self.results = previous_results
                stats = self._calculate_statistics()
                return (
                    len([r for r in self.results if r.success]),
                    len(self.results),
                    stats,
                )
        else:
            samples_to_process = all_samples

        self.logger.info(
            f"Evaluating {len(samples_to_process)} samples using {thread_num} threads"
        )

        success_count = 0
        total_count = 0

        # Progress bar for evaluation
        with tqdm(samples_to_process, desc="Evaluating", unit="sample") as pbar:
            # Use ThreadPoolExecutor for multi-threading
            with ThreadPoolExecutor(max_workers=thread_num) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(self.evaluate_one, sample): sample
                    for sample in samples_to_process
                }

                # Process completed tasks with timeout
                import concurrent.futures

                try:
                    # Use as_completed with timeout, but handle TimeoutError specifically
                    completed_futures = set()
                    for future in as_completed(future_to_sample):  # 24小时总超时
                        completed_futures.add(future)
                        sample = future_to_sample[future]
                        try:
                            result = future.result(timeout=360)  # 6分钟单任务超时

                            # Thread-safe operations
                            with self._lock:
                                self.results.append(result)
                                if result.success:
                                    success_count += 1
                                total_count += 1

                                # Update progress bar
                                current_rate = (
                                    success_count / total_count * 100
                                    if total_count > 0
                                    else 0
                                )
                                pbar.set_postfix(
                                    {
                                        "success_rate": f"{current_rate:.1f}%",
                                        "success": success_count,
                                        "total": total_count,
                                    }
                                )
                                pbar.update(1)

                        except concurrent.futures.TimeoutError:
                            self.logger.error(
                                f"Timeout processing sample {sample.get('sample_id', 'unknown')}"
                            )
                            # Create timeout error result
                            error_result = self.create_error_result(
                                sample=sample,
                                error_message="Timeout: Task exceeded maximum execution time",
                                execution_time=120.0,
                                raw_output="ERROR: Task timeout"
                            )
                            with self._lock:
                                self.results.append(error_result)
                                total_count += 1
                                pbar.update(1)

                        except Exception as e:
                            self.logger.error(
                                f"Error processing sample {sample.get('sample_id', 'unknown')}: {e}"
                            )
                            # Create error result
                            error_result = self.create_error_result(
                                sample=sample,
                                error_message=str(e),
                                execution_time=0.0,
                                raw_output="ERROR: Exception in thread execution"
                            )
                            with self._lock:
                                self.results.append(error_result)
                                total_count += 1
                                pbar.update(1)

                except concurrent.futures.TimeoutError:
                    # 处理as_completed的总体超时
                    self.logger.warning("Overall evaluation timeout - processing remaining tasks")

                # 处理未完成的任务（无论是否超时）
                for future, sample in future_to_sample.items():
                    if not future.done():
                        self.logger.warning(f"Cancelling unfinished task: {sample.get('sample_id', 'unknown')}")
                        future.cancel()
                        error_result = self.create_error_result(
                            sample=sample,
                            error_message="Cancelled: Task did not complete in time",
                            execution_time=0.0,
                            raw_output="ERROR: Task cancelled"
                        )
                        with self._lock:
                            self.results.append(error_result)
                            total_count += 1
                            pbar.update(1)

        # If resuming, merge with previous results
        if resume_from and previous_results:
            self.results = self.merge_results(previous_results, self.results)
            # Recalculate success count for merged results
            success_count = len([r for r in self.results if r.success])
            total_count = len(self.results)

        # Calculate detailed statistics
        stats = self._calculate_statistics()

        # Save results if configured
        if self.config.save_results:
            self._save_results(stats, resume_mode=bool(resume_from))

        mode_str = "RESUMED" if resume_from else "COMPLETED"
        self.logger.info(
            f"Evaluation {mode_str}: {success_count}/{total_count} ({success_count / total_count * 100:.2f}%)"
        )

        return success_count, total_count, stats

    def merge_results(
            self,
            previous_results: List[EvaluationResult],
            new_results: List[EvaluationResult],
    ) -> List[EvaluationResult]:
        """Merge previous successful results with new retry results."""
        # Create a map of sample_id to new results
        new_results_map = {result.sample_id: result for result in new_results}

        merged_results = []

        # Start with previous results, replacing with new ones where available
        for prev_result in previous_results:
            if prev_result.sample_id in new_results_map:
                # Use new result for this sample
                merged_results.append(new_results_map[prev_result.sample_id])
                self.logger.debug(f"Replaced result for sample {prev_result.sample_id}")
            else:
                # Keep previous successful result
                merged_results.append(prev_result)

        # Add any new results that weren't in previous results
        prev_sample_ids = {result.sample_id for result in previous_results}
        for new_result in new_results:
            if new_result.sample_id not in prev_sample_ids:
                merged_results.append(new_result)
                self.logger.debug(f"Added new result for sample {new_result.sample_id}")

        self.logger.info(
            f"Merged results: {len(merged_results)} total, {len(new_results)} newly processed"
        )
        return merged_results

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

        # Domain/category breakdown
        domain_stats = {}
        for result in self.results:
            sample_parts = result.sample_id.split("_")
            if len(sample_parts) >= 2:
                domain = sample_parts[0]
                category = sample_parts[1]

                if domain not in domain_stats:
                    domain_stats[domain] = {"total": 0, "success": 0, "categories": {}}

                domain_stats[domain]["total"] += 1
                if result.success:
                    domain_stats[domain]["success"] += 1

                if category not in domain_stats[domain]["categories"]:
                    domain_stats[domain]["categories"][category] = {
                        "total": 0,
                        "success": 0,
                    }

                domain_stats[domain]["categories"][category]["total"] += 1
                if result.success:
                    domain_stats[domain]["categories"][category]["success"] += 1

        # Calculate success rates for domains/categories
        for domain in domain_stats:
            if domain_stats[domain]["total"] > 0:
                domain_stats[domain]["success_rate"] = (
                        domain_stats[domain]["success"]
                        / domain_stats[domain]["total"]
                        * 100
                )

            for category in domain_stats[domain]["categories"]:
                cat_data = domain_stats[domain]["categories"][category]
                if cat_data["total"] > 0:
                    cat_data["success_rate"] = (
                            cat_data["success"] / cat_data["total"] * 100
                    )

        return {
            "total_samples": total_samples,
            "success_count": success_count,
            "success_rate": success_rate,
            "error_count": len(error_results),
            "error_rate": error_rate,
            "avg_execution_time": avg_execution_time,
            "domain_stats": domain_stats,
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _save_results(self, stats: Dict[str, Any], resume_mode: bool = False):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_resumed" if resume_mode else ""

        # Save detailed results
        results_file = os.path.join(
            self.config.output_dir, f"evaluation_results_{timestamp}{mode_suffix}.json"
        )
        detailed_results = {
            "config": {
                "root_dir": self.config.root_dir,
                "model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "max_samples": self.config.max_samples,
                "resume_mode": resume_mode,
            },
            "statistics": stats,
            "detailed_results": [
                {f.name: getattr(r, f.name) for f in fields(r)} for r in self.results
            ],
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        # Save summary stats
        summary_file = os.path.join(
            self.config.output_dir, f"evaluation_summary_{timestamp}{mode_suffix}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Summary saved to: {summary_file}")
