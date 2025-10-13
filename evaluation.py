import argparse
import logging

from dataclasses import dataclass
from evaluator.grounding import GroundingEvaluator
from evaluator.action_prediction import ActionPredictionEvaluator
from evaluator.action_prediction_a11y import ActionPredictionA11yEvaluator
from evaluator.screen_parsing import ScreenParsingEvaluator
from config import EvaluationConfig

from models.gpt import GPT_predictor
import threading

lock = threading.Lock()
LOCAL_MODEL = None


class ModelFactory:
    """Factory class for loading different models."""

    @staticmethod
    def create_model(model_type: str, model_name: str, result_path: str = None, api_url: str = None):
        """Create a model instance based on type."""
        if model_type == "qwen2.5_vl_7b":
            return ModelFactory._load_qwen_model(model_name, api_url)
        elif model_type.lower() == "gpt":
            return ModelFactory._load_gpt_model(model_name)
        elif model_type.lower() == "gui_actor":
            return ModelFactory._load_gui_actor_model(model_name, api_url)
        elif model_type.lower() == "uground":
            return ModelFactory._load_uground_model(model_name, api_url)
        elif model_type.lower() == "ui_tars" or model_type.lower() == "uitars":
            return ModelFactory._load_ui_tars_model(model_name, api_url)
        elif model_type.lower() == "aguvis" or model_type.lower() == "aguvis_7b_720p":
            return ModelFactory._load_aguvis_model(model_name, api_url)
        elif model_type.lower() == "omniparser":
            return ModelFactory._load_omniparser_model(model_name, api_url)
        elif model_type.lower() == "mock":
            return ModelFactory._load_mock_model(model_name, result_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _load_qwen_model(model_name: str, api_url: str = None):
        """Load Qwen 2.5 VL model."""
        try:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location(
                "qwen2_5_vl_7b", "models/qwen2.5_vl_7b.py"
            )
            qwen_module = importlib.util.module_from_spec(spec)
            sys.modules["qwen2_5_vl_7b"] = qwen_module
            spec.loader.exec_module(qwen_module)

            if api_url is None:
                api_url = "http://20.91.255.106:19806/v1"
            
            return qwen_module.Qwen25VL7B(api_url=api_url, model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen model: {e}")

    @staticmethod
    def _load_gpt_model(model_name: str):
        """Load GPT model."""
        return GPT_predictor(model_name)

    @staticmethod
    def _load_gui_actor_model(model_name: str, api_url: str = None):
        """Load GUI Actor model."""
        try:
            from models.gui_actor import GuiActor

            if api_url is None:
                api_url = "http://20.91.255.106:19806/evaluate"

            return GuiActor(api_url=api_url, model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load GUI Actor model: {e}")

    @staticmethod
    def _load_uground_model(model_name: str, api_url: str = None):
        """Load UGround model."""
        try:
            from models.uground import UGround

            if api_url is None:
                api_url = "http://20.91.255.106:19806/v1"

            return UGround(model_name=model_name, api_url=api_url)
        except Exception as e:
            raise RuntimeError(f"Failed to load UGround model: {e}")

    @staticmethod
    def _load_ui_tars_model(model_name: str, api_url: str = None):
        """Load UI-TARS model."""
        try:
            from models.ui_tars import UITars

            if api_url is None:
                api_url = "http://20.91.255.106:19806/v1"

            return UITars(model_name=model_name, api_url=api_url)
        except Exception as e:
            raise RuntimeError(f"Failed to load UI-TARS model: {e}")

    @staticmethod
    def _load_aguvis_model(model_name: str, api_url: str = None):
        """Load Aguvis-7B-720P model."""
        try:
            from models.aguvis_7b_720p import Aguvis7B720P

            if api_url is None:
                api_url = "http://20.91.255.106:19806/evaluate"

            return Aguvis7B720P(api_url=api_url, model_name=model_name, use_smart_resize=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Aguvis model: {e}")

    @staticmethod
    def _load_omniparser_model(model_name: str, api_url: str = None):
        """Load Omniparser model."""
        try:
            from models.omniparser import OmniparserScreenParsing, OmniparserGrounding
            from urllib.parse import urlparse

            if api_url:
                parsed = urlparse(api_url)
                host = parsed.hostname or '20.91.255.106'
                port = parsed.port or 19806
            else:
                host = '20.91.255.106'
                port = 19806

            config = {
                'som_model_path': '../../weights/icon_detect/model.pt',
                'caption_model_name': 'florence2',
                'caption_model_path': '../../weights/icon_caption_florence',
                'device': 'cuda',
                'BOX_TRESHOLD': 0.05,
                'host': host,
                'port': port
            }

            if "screen" in model_name.lower() or "parsing" in model_name.lower():
                return OmniparserScreenParsing(model_name, config)
            else:
                return OmniparserGrounding(model_name, config)

        except Exception as e:
            raise RuntimeError(f"Failed to load Omniparser model: {e}")

    @staticmethod
    def _load_mock_model(model_name: str, result_path: str):
        """Load Mock model."""
        lock.acquire()
        try:
            from models.mock_model import MockModel

            global LOCAL_MODEL
            if not LOCAL_MODEL:
                LOCAL_MODEL = MockModel(model_name, result_path)
            return LOCAL_MODEL

        except Exception as e:
            raise RuntimeError(f"Failed to load Mock model: {e}")

        finally:
            lock.release()


def create_config_from_args(args) -> EvaluationConfig:
    """Create configuration from command line arguments."""
    return EvaluationConfig(
        root_dir=args.root_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        eval_type=args.type,
        max_samples=args.max_samples,
        save_results=not args.no_save,
        output_dir=args.output_dir,
        log_level=args.log_level,
        result_path=args.result_path,
        api_url=args.api_url,
    )


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="GUI Grounding Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to dataset root directory (e.g., data/grounding)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to previous evaluation results file to resume from (will only re-run error cases)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt",
        choices=["qwen2.5_vl_7b", "gpt", "gui_actor", "uground", "ui_tars", "aguvis", "omniparser", "mock"],
        help="Model type",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="Path to result file",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=None,
        help="API URL for model inference (e.g., http://20.91.255.106:19806/v1)",
    )
    # Evaluation configuration
    parser.add_argument(
        "--type",
        type=str,
        default="grounding",
        choices=["grounding", "action_prediction", "action_prediction_a11y", "screen_parsing"],
        help="Evaluation type",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of threads for parallel evaluation (default: 5)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--no_save", action="store_true", help="Do not save detailed results"
    )

    # Logging configuration
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    try:
        config = create_config_from_args(args)

        main_logger = logging.getLogger("main")
        if not main_logger.handlers:
            main_logger.setLevel(getattr(logging, config.log_level))
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            main_logger.addHandler(handler)
            main_logger.propagate = False
        logger = main_logger

        logger.info("=== Evaluation ===")
        logger.info(f"Dataset: {config.root_dir}")
        logger.info(f"Model: {config.model_name} ({config.model_type})")
        logger.info(f"Max samples: {config.max_samples or 'All'}")
        logger.info(f"Threads: {args.threads}")
        logger.info(f"Resume from: {args.resume_from or 'None (fresh start)'}")
        logger.info(f"Output: {config.output_dir}")
        logger.info("=" * 40)

        logger.info("Creating evaluator...")
        if config.eval_type == "grounding":
            evaluator = GroundingEvaluator(config)
        elif config.eval_type == "action_prediction":
            evaluator = ActionPredictionEvaluator(config)
        elif config.eval_type == "action_prediction_a11y":
            evaluator = ActionPredictionA11yEvaluator(config)
        elif config.eval_type == "screen_parsing":
            evaluator = ScreenParsingEvaluator(config)
        else:
            raise ValueError(f"Unsupported evaluation type: {config.eval_type}")
        logger.info("Evaluator created successfully!")

        success, total, stats = evaluator.evaluate(
            thread_num=args.threads, resume_from=args.resume_from
        )

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {total}")
        print(f"Successful predictions: {success}")
        print(f"Success rate: {success / total * 100:.2f}%")
        print(f"Error rate: {stats.get('error_rate', 0):.2f}%")
        print(f"Average execution time: {stats.get('avg_execution_time', 0):.3f}s")

        if "domain_stats" in stats:
            print(f"\nDomain Breakdown:")
            for domain, domain_data in stats["domain_stats"].items():
                print(
                    f"  {domain}: {domain_data['success']}/{domain_data['total']} ({domain_data.get('success_rate', 0):.1f}%)"
                )

                for category, cat_data in domain_data["categories"].items():
                    print(
                        f"    {category}: {cat_data['success']}/{cat_data['total']} ({cat_data.get('success_rate', 0):.1f}%)"
                    )

        print("=" * 60)

        if config.save_results:
            print(f"Detailed results saved to: {config.output_dir}/")

        return 0

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
