from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    root_dir: str
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_type: str = "qwen2.5_vl_7b"
    eval_type: str = "grounding"
    max_samples: Optional[int] = None
    save_results: bool = True
    output_dir: str = "results"
    log_level: str = "INFO"
    result_path: str = None
    api_url: str = None
