import json
import os
import shutil
import re
import math
import argparse
from prompt_action_prediction import (
    ACTION_PREDICTION_USER_PROMPT_QWEN, 
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_PPT,
    SUPPORTED_ACTIONS_EXCEL_A11Y,
    SUPPORTED_ACTIONS_WORD_A11Y,
    SUPPORTED_ACTIONS_PPT_A11Y,
    ACTION_PREDICTION_USER_PROMPT_QWEN_A11Y
)
from prompt_screen_parsing import SCREEN_PARSING_SYS_PROMPT
from prompt_grounding import GROUNDING_USER_PROMPT_QWEN

# Try to import PIL and other dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image processing will be disabled.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")

# Try to import qwen_vl_utils
try:
    from qwen_vl_utils import smart_resize
    QWEN_VL_UTILS_AVAILABLE = True
    print("✅ Using qwen_vl_utils.smart_resize")
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("⚠️ qwen_vl_utils not available, using fallback smart_resize implementation")


class Convertor:
    """Convertor class with improved functionality."""
    
    def __init__(self):
        """Initialize the convertor with domain-specific supported actions."""
        self.domain_actions = {
            "excel": SUPPORTED_ACTIONS_EXCEL,
            "word": SUPPORTED_ACTIONS_WORD,
            "ppt": SUPPORTED_ACTIONS_PPT
        }

        self.domain_actions_a11y = {
            "excel": SUPPORTED_ACTIONS_EXCEL_A11Y,
            "word": SUPPORTED_ACTIONS_WORD_A11Y,
            "ppt": SUPPORTED_ACTIONS_PPT_A11Y
        }

        self.word_case_number = 0
        self.excel_case_number = 0
        self.ppt_case_number = 0

    def smart_resize_fallback(self, height: int, width: int, factor: int = 28, max_pixels: int = 999999):
        """
        Fallback implementation of smart_resize when qwen_vl_utils is not available.
        
        Args:
            height (int): Original height of the image.
            width (int): Original width of the image.
            factor (int): The dimensions must be divisible by this factor.
            max_pixels (int): The maximum number of pixels for the resized image.
            
        Returns:
            Tuple[int, int]: The new (height, width).
        """
        original_pixels = height * width
        
        # Calculate the scaling ratio if the image exceeds max_pixels
        if original_pixels > max_pixels:
            scale_ratio = math.sqrt(max_pixels / original_pixels)
            new_height = int(round(height * scale_ratio))
            new_width = int(round(width * scale_ratio))
        else:
            new_height = height
            new_width = width
            
        # Ensure the new dimensions are divisible by the factor
        final_height = int(round(new_height / factor) * factor)
        final_width = int(round(new_width / factor) * factor)
        
        # Handle cases where rounding might result in zero
        if final_height == 0:
            final_height = factor
        if final_width == 0:
            final_width = factor
            
        return final_height, final_width

    def get_supported_actions(self, domain):
        """Get supported actions based on domain."""
        domain_lower = domain.lower()
        if domain_lower in self.domain_actions:
            return self.domain_actions[domain_lower]
        else:
            print(f"Warning: Unknown domain '{domain}', using Excel actions as default")
            return self.domain_actions["excel"]

    def get_supported_actions_a11y(self, domain):
        """Get supported actions based on domain."""
        domain_lower = domain.lower()
        if domain_lower in self.domain_actions_a11y:
            return self.domain_actions_a11y[domain_lower]
        else:
            print(f"Warning: Unknown domain '{domain}', using Excel actions as default")
            return self.domain_actions_a11y["excel"]

    def extract_and_scale_coordinates(self, text, scale_w, scale_h):
        """
        Extract coordinates from text and scale them according to image resize.
        
        Args:
            text (str): Text containing coordinates
            scale_w (float): Width scaling factor
            scale_h (float): Height scaling factor
            
        Returns:
            str: Text with scaled coordinates
        """
        # Pattern for different coordinate formats
        patterns = [
            # Standard coordinate: [x, y]
            (r'"coordinate":\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', 'coordinate'),
            # start_coordinate: [x, y]
            (r'"start_coordinate":\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', 'start_coordinate'),
            # end_coordinate: [x, y]
            (r'"end_coordinate":\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', 'end_coordinate'),
        ]
        
        updated_text = text
        
        for pattern, coord_type in patterns:
            def replace_coord(match):
                x = float(match.group(1))
                y = float(match.group(2))
                
                # Scale coordinates
                new_x = x * scale_w
                new_y = y * scale_h
                
                # Return the scaled coordinate
                return f'"{coord_type}": [{new_x:.1f}, {new_y:.1f}]'
            
            updated_text = re.sub(pattern, replace_coord, updated_text)
        
        return updated_text

    def load_data(self, task_type, root_dir, success_or_fail):
        """Load data from dataset directory and construct previous actions for each sample."""
        root_path = os.path.join(root_dir)
        sample_count = 0
        all_samples = []

        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Dataset directory not found: {root_path}")

        data_path = os.path.join(root_path, "data")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        # Process /data directory
        domain_folders = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]

        if TQDM_AVAILABLE:
            domain_iterator = tqdm(domain_folders, desc="Processing domains", unit="domain")
        else:
            domain_iterator = domain_folders
            
        for domain in domain_iterator:
            domain_path = os.path.join(data_path, domain)
            category_folders = [
                c
                for c in os.listdir(domain_path)
                if os.path.isdir(os.path.join(domain_path, c))
            ]

            if TQDM_AVAILABLE:
                category_iterator = tqdm(category_folders, desc=f"Processing {domain} categories", unit="category", leave=False)
            else:
                category_iterator = category_folders
                
            for category in category_iterator:
                category_path = os.path.join(domain_path, category, success_or_fail)
                if not os.path.exists(category_path):
                    continue

                jsonl_files = [
                    f for f in os.listdir(category_path) if f.endswith(".jsonl")
                ]

                if TQDM_AVAILABLE and jsonl_files:
                    jsonl_iterator = tqdm(jsonl_files, desc=f"Processing {category} files", unit="file", leave=False)
                else:
                    jsonl_iterator = jsonl_files
                    
                for jsonl_file in jsonl_iterator:
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
                                    continue

                        # First pass: determine which steps are valid (not filtered out)
                        valid_steps = []
                        
                        if TQDM_AVAILABLE and len(all_steps) > 10:
                            step_iterator = tqdm(enumerate(all_steps), desc=f"Filtering {jsonl_file} steps", total=len(all_steps), unit="step", leave=False)
                        else:
                            step_iterator = enumerate(all_steps)
                            
                        for i, step_info in step_iterator:
                            data = step_info["data"]
                            
                            # Check if this sample has the required task type
                            if task_type in data["step"]["tags"]:
                                # Check if image exists - handle different path formats
                                screenshot_path = data["step"]["screenshot_clean"]
                                
                                # Try different path construction methods
                                possible_paths = [
                                    # Method 1: Direct path from root
                                    os.path.join(root_path, screenshot_path),
                                    # Method 2: Standard structure: root/image/domain/category/file
                                    os.path.join(root_path, "image", domain, category, screenshot_path),
                                    # Method 3: Extract filename and use standard structure
                                    os.path.join(root_path, "image", domain, category, os.path.basename(screenshot_path))
                                ]
                                
                                clean_img_path = None
                                for i, path in enumerate(possible_paths):
                                    if os.path.exists(path):
                                        clean_img_path = path
                                        break
                                
                                # Debug information for path resolution
                                if clean_img_path is None and TQDM_AVAILABLE:
                                    tqdm.write(f"⚠️ Image not found for {screenshot_path}")
                                    tqdm.write(f"   Tried paths: {possible_paths}")
                                elif clean_img_path is None:
                                    print(f"⚠️ Image not found for {screenshot_path}")
                                    print(f"   Tried paths: {possible_paths}")
                                if clean_img_path is None:
                                    continue
                                
                                # Check if this step would be filtered out due to empty function
                                action_data = data["step"]["action"]
                                try:
                                    if isinstance(action_data, str):
                                        action_data = json.loads(action_data)
                                    function_name = action_data.get("function", "")

                                    # Skip if empty function and not FINISH status
                                    if not function_name and status == "OVERALL_FINISH":
                                        continue
                                    
                                    # Apply the same filter logic as in convert methods
                                    status = data["step"]["status"]
                                    if status == "OVERALL_FINISH":
                                        status = "FINISH"
                                    elif status == "FINISH":
                                        status = "CONTINUE"
                                    
                                    # This step is valid, add to valid_steps
                                    valid_steps.append({
                                        "original_index": i,
                                        "step_info": step_info,
                                        "data": data,
                                        "status": status,
                                        "clean_img_path": clean_img_path
                                    })
                                    
                                except Exception as e:
                                    # Skip steps with invalid action data
                                    continue

                        # Second pass: process valid steps and construct history from valid steps only
                        for valid_step_index, valid_step in enumerate(valid_steps):
                            step_info = valid_step["step_info"]
                            data = valid_step["data"]
                            status = valid_step["status"]
                            clean_img_path = valid_step["clean_img_path"]
                            line_num = step_info["line_num"]

                            # Create sample ID
                            sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{line_num}"

                            # Build annotated image path using same logic
                            annotated_screenshot_path = data["step"]["screenshot_annotated"]
                            possible_annotated_paths = [
                                os.path.join(root_path, annotated_screenshot_path),
                                os.path.join(root_path, "image", domain, category, annotated_screenshot_path),
                                os.path.join(root_path, "image", domain, category, os.path.basename(annotated_screenshot_path))
                            ]
                            
                            annotated_img_path = None
                            for path in possible_annotated_paths:
                                if os.path.exists(path):
                                    annotated_img_path = path
                                    break
                            
                            # Use clean_img_path as fallback if annotated not found
                            if annotated_img_path is None:
                                annotated_img_path = clean_img_path

                            # Construct previous actions from earlier VALID steps only
                            previous_actions = []
                            for j in range(valid_step_index):
                                prev_valid_step = valid_steps[j]
                                prev_data = prev_valid_step["data"]
                                prev_thought = prev_data["step"]["thought"]
                                previous_actions.append(
                                    f"Step {j+1}: {prev_thought}"
                                )

                            sample = {
                                "sample_id": sample_id,
                                "request": data["request"],
                                "screenshot_clean": clean_img_path,
                                "screenshot_annotated": annotated_img_path,
                                "thought": "Step " + str(valid_step_index + 1) + ": " + data["step"]["thought"],
                                "action": data["step"]["action"],
                                "control_infos": data["step"]["control_infos"],
                                "status": status,
                                "domain": domain,
                                "category": category,
                                "previous_actions": previous_actions,
                                "step_index": valid_step_index + 1,
                            }

                            if domain == "word":
                                self.word_case_number += 1
                                if self.word_case_number > 34000:
                                    continue
                            elif domain == "excel":
                                self.excel_case_number += 1
                                if self.excel_case_number > 34000:
                                    continue
                            elif domain == "ppt":
                                self.ppt_case_number += 1
                                if self.ppt_case_number > 34000:
                                    continue

                            yield sample
                            sample_count += 1
                            if not TQDM_AVAILABLE and sample_count % 1000 == 0:
                                print(f"Loaded {sample_count} samples so far")

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
                        continue

        print(f"Loaded {sample_count} samples total")

    def convert_to_action_prediction(self, root_dir, success_or_fail, output_dir=None):
        """Convert data to action prediction training format using ACTION_PREDICTION_USER_PROMPT_QWEN."""
        data_generator = self.load_data(
            task_type="action_prediction", 
            root_dir=root_dir, 
            success_or_fail=success_or_fail
        )
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory created: {output_dir}")
        
        training_data = []
        domain_stats = {}  # Track statistics by domain
        
        # Convert generator to list to get total count for progress bar
        print("Loading samples...")
        samples_list = list(data_generator)
        
        if TQDM_AVAILABLE:
            sample_iterator = tqdm(samples_list, desc="Converting samples", unit="sample")
        else:
            sample_iterator = samples_list
            
        for sample in sample_iterator:
            domain = sample["domain"]
            
            # Track domain statistics
            if domain not in domain_stats:
                domain_stats[domain] = 0
            domain_stats[domain] += 1
            
            # Get domain-specific supported actions
            supported_actions = self.get_supported_actions(domain)
            
            # Format history of actions
            if sample["previous_actions"]:
                history = "\n".join(sample["previous_actions"])
            else:
                history = ""
            
            # Create the human message with the prompt using domain-specific actions
            human_message = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
                instruction=sample["request"],
                history=history,
                actions=supported_actions
            )
            
            # Format the action as a tool call
            action_data = sample["action"]
            try:
                # Parse the action if it's a string
                if isinstance(action_data, str):
                    action_data = json.loads(action_data)
                
                # Extract function name and arguments from action data
                function_name = ""
                args = {}
                
                if "function" in action_data:
                    function_name = action_data["function"]
                    # Copy all other fields as arguments, excluding 'action'
                    args = action_data["args"]
                
                # Filter out actions with empty function name and status not OVERALL_FINISH
                if not function_name and sample["status"] != "FINISH":
                    print(f"Skipping sample {sample['sample_id']}: empty function and status is not FINISH")
                    continue

                if action_data["function"] == "drag":
                    args["start_coordinate"] = [args["start_x"], args["start_y"]]
                    args["end_coordinate"] = [args["end_x"], args["end_y"]]

                    del args["start_x"]
                    del args["start_y"]
                    del args["end_x"]
                    del args["end_y"]
                else:
                    args["coordinate"] = [action_data["coordinate_x"], action_data["coordinate_y"]]
                    if "x" in args:
                        del args["x"]
                    if "y" in args:
                        del args["y"]
                
                # Create the tool call format
                tool_call = {
                    "function": function_name,
                    "args": args,
                    "status": sample["status"]
                }
                
                # Format the GPT response with thought and tool call
                thought_text = sample["thought"]
                gpt_response = f"<think> {thought_text} </think><tool_call>\n{json.dumps(tool_call, indent=4)}\n</tool_call>"
                
            except Exception as e:
                print(f"Error processing action for sample {sample['sample_id']}: {e}")
                continue
            
            # Handle image paths and copying
            sample_id = sample["sample_id"]
            image_paths = []
            
            if output_dir:
                # Extract jsonl filename from sample_id (format: domain_category_jsonlfile_linenum)
                parts = sample_id.split('_')
                if len(parts) >= 4:
                    # Reconstruct jsonl filename (everything except domain, category, and line number)
                    jsonl_filename = '_'.join(parts[-4:-1])  # Remove domain, category, and line number
                else:
                    jsonl_filename = "unknown"
                
                # Create images directory structure: output_dir/images/[jsonl_filename]/
                images_dir = os.path.join(output_dir, "images", jsonl_filename)
                os.makedirs(images_dir, exist_ok=True)
                
                # Copy images and update paths
                if os.path.exists(sample["screenshot_clean"]):
                    clean_filename = os.path.basename(sample["screenshot_clean"])
                    clean_dest = os.path.join(images_dir, clean_filename)
                    shutil.copy2(sample["screenshot_clean"], clean_dest)
                    image_paths.append(f"images/{jsonl_filename}/{clean_filename}")
                    print(f"Copied image for {domain}: {clean_dest}")
                else:
                    print(f"Warning: Image not found for {domain}: {sample['screenshot_clean']}")
                    image_paths.append(sample["screenshot_clean"])
            else:
                # Use original paths
                image_paths.append(sample["screenshot_clean"])
            
            # Create the training sample
            training_sample = {
                "id": sample_id,
                "images": image_paths,
                "conversation": [
                    {
                        "from": "human",
                        "value": f"<image>\n{human_message}"
                    },
                    {
                        "from": "gpt", 
                        "value": gpt_response
                    }
                ],
                "action": tool_call
            }
            
            training_data.append(training_sample)
        
        # Print domain statistics
        print(f"\n=== Domain Statistics ===")
        for domain, count in domain_stats.items():
            print(f"{domain}: {count} samples")
            print(f"  Using actions: {type(self.get_supported_actions(domain)).__name__}")
        
        return training_data

    def convert_to_action_prediction_resize(self, root_dir, success_or_fail, output_dir, max_pixels=999999, factor=28):
        """
        Convert data to action prediction training format with image resizing and coordinate scaling.
        
        Args:
            root_dir (str): Root directory of the dataset
            success_or_fail (str): Which subset to process ('success' or 'fail')
            output_dir (str): Output directory for processed data and images
            max_pixels (int): Maximum number of pixels for resized images
            factor (int): Factor that image dimensions must be divisible by
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for image processing. Please install it: pip install Pillow")
        
        data_generator = self.load_data(
            task_type="action_prediction", 
            root_dir=root_dir, 
            success_or_fail=success_or_fail
        )
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        processed_images_dir = os.path.join(output_dir, "images")
        os.makedirs(processed_images_dir, exist_ok=True)
        
        print(f"🖼️  Resized images will be saved in: {processed_images_dir}")
        print(f"📄 Training data will be saved in: {output_dir}")
        
        training_data = []
        domain_stats = {}
        processing_stats = {
             "total_processed": 0,
             "images_resized": 0,
             "coordinates_scaled": 0,
             "skipped_no_coordinates": 0,
             "skipped_no_image": 0,
             "skipped_empty_function": 0
         }
        
        samples_list = list(data_generator)
        print(f"Loaded {len(samples_list)} samples")
        iterator = tqdm(samples_list, desc="Processing samples", unit="sample") if TQDM_AVAILABLE else samples_list
        
        for sample in iterator:
            domain = sample["domain"]
            sample_id = sample["sample_id"]
            
            # Track domain statistics
            if domain not in domain_stats:
                domain_stats[domain] = 0
            domain_stats[domain] += 1
            
            # Get domain-specific supported actions
            supported_actions = self.get_supported_actions(domain)
            
            # Format history of actions
            if sample["previous_actions"]:
                history = "\n".join(sample["previous_actions"])
            else:
                history = ""
            
            # Create the human message with the prompt using domain-specific actions
            human_message = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
                instruction=sample["request"],
                history=history,
                actions=supported_actions
            )
            
            # Format the action as a tool call
            action_data = sample["action"]
            try:
                # Parse the action if it's a string
                if isinstance(action_data, str):
                    action_data = json.loads(action_data)
                
                # Extract function name and arguments from action data
                function_name = ""
                args = {}
                
                if "function" in action_data:
                    function_name = action_data["function"]
                    # Copy all other fields as arguments, excluding 'action'
                    args = action_data["args"]
                
                # Filter out actions with empty function name and status not OVERALL_FINISH
                if not function_name and sample["status"] != "FINISH":
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Skipping sample {sample_id}: empty function and status is not FINISH")
                    else:
                        print(f"Skipping sample {sample_id}: empty function and status is not FINISH")
                    processing_stats["skipped_empty_function"] = processing_stats.get("skipped_empty_function", 0) + 1
                    continue

                if action_data["function"] == "drag":
                    args["start_coordinate"] = [args["start_x"], args["start_y"]]
                    args["end_coordinate"] = [args["end_x"], args["end_y"]]

                    del args["start_x"]
                    del args["start_y"]
                    del args["end_x"]
                    del args["end_y"]
                else:
                    args["coordinate"] = [action_data["coordinate_x"], action_data["coordinate_y"]]
                    if "x" in args:
                        del args["x"]
                    if "y" in args:
                        del args["y"]
                
                # Create the tool call format
                tool_call = {
                    "function": function_name,
                    "args": args,
                    "status": sample["status"]
                }
                
                # Convert tool call to JSON string for coordinate processing
                tool_call_json = json.dumps(tool_call, indent=4)
                
                # Format the GPT response with thought and tool call
                thought_text = sample["thought"]
                gpt_response = f"<tool_call>\n{tool_call_json}\n</tool_call>"
                 
            except Exception as e:
                 if TQDM_AVAILABLE:
                     tqdm.write(f"Error processing action for sample {sample_id}: {e}")
                 else:
                     print(f"Error processing action for sample {sample_id}: {e}")
                 continue
            
            # Process image and coordinates
            original_image_path = sample["screenshot_clean"]
            
            if not os.path.exists(original_image_path):
                if TQDM_AVAILABLE:
                    tqdm.write(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                else:
                    print(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                processing_stats["skipped_no_image"] += 1
                continue
            
            # Check if the response contains coordinates that need scaling
            has_coordinates = any(coord_type in tool_call_json for coord_type in ["coordinate", "start_coordinate", "end_coordinate"])
            
            if not has_coordinates:
                # No coordinates to process, just copy image and save sample
                try:
                    # Extract jsonl filename from sample_id
                    parts = sample_id.split('_')
                    if len(parts) >= 4:
                        jsonl_filename = '_'.join(parts[-4:-1])
                    else:
                        jsonl_filename = "unknown"
                    
                    images_dir = os.path.join(output_dir, "images", jsonl_filename)
                    os.makedirs(images_dir, exist_ok=True)
                    
                    clean_filename = os.path.basename(original_image_path)
                    clean_dest = os.path.join(images_dir, clean_filename)
                    shutil.copy2(original_image_path, clean_dest)
                    
                    image_paths = [f"images/{jsonl_filename}/{clean_filename}"]
                    processing_stats["skipped_no_coordinates"] += 1
                    
                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Error copying image for {sample_id}: {e}")
                    else:
                        print(f"Error copying image for {sample_id}: {e}")
                    continue
            else:
                # Process image resizing and coordinate scaling
                try:
                    with Image.open(original_image_path) as img:
                        original_width, original_height = img.size
                        
                        # Calculate new dimensions using smart_resize
                        if QWEN_VL_UTILS_AVAILABLE:
                            new_height, new_width = smart_resize(
                                height=original_height,
                                width=original_width,
                                factor=factor,
                                max_pixels=max_pixels
                            )
                        else:
                            new_height, new_width = self.smart_resize_fallback(
                                height=original_height,
                                width=original_width,
                                factor=factor,
                                max_pixels=max_pixels
                            )
                        
                        # Calculate scaling factors
                        scale_w = new_width / original_width
                        scale_h = new_height / original_height
                        
                                                 # Resize image and save
                        resized_img = img.resize((new_width, new_height))
                        
                        # Extract jsonl filename and create images directory
                        parts = sample_id.split('_')
                        if len(parts) >= 4:
                            jsonl_filename = '_'.join(parts[-4:-1])
                        else:
                            jsonl_filename = "unknown"
                        
                        images_dir = os.path.join(output_dir, "images", jsonl_filename)
                        os.makedirs(images_dir, exist_ok=True)
                        
                        clean_filename = os.path.basename(original_image_path)
                        clean_dest = os.path.join(images_dir, clean_filename)
                        resized_img.save(clean_dest)
                        
                        image_paths = [f"images/{jsonl_filename}/{clean_filename}"]
                        
                        # Scale coordinates in the GPT response
                        updated_gpt_response = self.extract_and_scale_coordinates(gpt_response, scale_w, scale_h)
                        gpt_response = updated_gpt_response

                        if action_data["rectangle"]:
                            bbox = [
                                action_data["rectangle"]["left"] * scale_w,
                                action_data["rectangle"]["top"] * scale_h,
                                action_data["rectangle"]["right"] * scale_w,
                                action_data["rectangle"]["bottom"] * scale_h
                            ]
                        else:
                            bbox = None
                        
                        processing_stats["images_resized"] += 1
                        processing_stats["coordinates_scaled"] += 1
                        
                        if TQDM_AVAILABLE:
                            tqdm.write(f"✅ Processed {sample_id}: {original_width}x{original_height} → {new_width}x{new_height}")
                        
                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"❌ Error processing image for {sample_id}: {e}")
                    else:
                        print(f"❌ Error processing image for {sample_id}: {e}")
                    continue
            
            # Create the training sample
            training_sample = {
                "id": sample_id,
                "images": image_paths,
                "conversation": [
                    {
                        "from": "human",
                        "value": f"<image>\n{human_message}"
                    },
                    {
                        "from": "gpt", 
                        "value": gpt_response
                    }
                ],
                "reward": 1 if success_or_fail == "success" else 0,
                "bbox": bbox
            }
            
            training_data.append(training_sample)
            processing_stats["total_processed"] += 1
        
            # Print statistics
        print(f"\n=== Processing Statistics ===")
        print(f"Total samples processed: {processing_stats['total_processed']}")
        print(f"Images resized: {processing_stats['images_resized']}")
        print(f"Coordinates scaled: {processing_stats['coordinates_scaled']}")
        print(f"Skipped (no coordinates): {processing_stats['skipped_no_coordinates']}")
        print(f"Skipped (no image): {processing_stats['skipped_no_image']}")
        print(f"Skipped (empty function): {processing_stats['skipped_empty_function']}")
        
        print(f"\n=== Domain Statistics ===")
        for domain, count in domain_stats.items():
            print(f"{domain}: {count} samples")
        
        return training_data

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

    def scale_a11y_rectangles(self, a11y, scale_w, scale_h):
        """
        Scale all rectangles in a11y information according to image resize factors.
        
        Args:
            a11y: The a11y information with control rectangles
            scale_w (float): Width scaling factor
            scale_h (float): Height scaling factor
            
        Returns:
            dict: a11y information with scaled rectangles
        """
        # Make a deep copy to avoid modifying the original
        import copy
        scaled_a11y = copy.deepcopy(a11y)
        
        # Scale application window rectangle
        if "application_windows_info" in scaled_a11y and "control_rect" in scaled_a11y["application_windows_info"]:
            app_rect = scaled_a11y["application_windows_info"]["control_rect"]
            scaled_a11y["application_windows_info"]["control_rect"] = [
                int(app_rect[0] * scale_w),
                int(app_rect[1] * scale_h),
                int(app_rect[2] * scale_w),
                int(app_rect[3] * scale_h)
            ]
        
        # Scale all control rectangles
        if "uia_controls_info" in scaled_a11y:
            for control_info in scaled_a11y["uia_controls_info"]:
                if "control_rect" in control_info:
                    rect = control_info["control_rect"]
                    control_info["control_rect"] = [
                        int(rect[0] * scale_w),
                        int(rect[1] * scale_h),
                        int(rect[2] * scale_w),
                        int(rect[3] * scale_h)
                    ]
        
        return scaled_a11y

    def convert_to_action_prediction_resize_a11y(self, root_dir, success_or_fail, output_dir, max_pixels=999999, factor=28):

        
        """
        Convert data to action prediction training format with a11y information.
        
        Args:
            root_dir (str): Root directory of the dataset
            success_or_fail (str): Which subset to process ('success' or 'fail')
            output_dir (str): Output directory for processed data and images
            max_pixels (int): Maximum number of pixels for resized images
            factor (int): Factor that image dimensions must be divisible by
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for image processing. Please install it: pip install Pillow")
        
        data_generator = self.load_data(
            task_type="action_prediction", 
            root_dir=root_dir, 
            success_or_fail=success_or_fail
        )
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        processed_images_dir = os.path.join(output_dir, "images")
        os.makedirs(processed_images_dir, exist_ok=True)
        
        print(f"🖼️  Resized images will be saved in: {processed_images_dir}")
        print(f"📄 Training data will be saved in: {output_dir}")
        
        training_data = []
        domain_stats = {}
        processing_stats = {
             "total_processed": 0,
             "images_resized": 0,
             "coordinates_scaled": 0,
             "skipped_no_coordinates": 0,
             "skipped_no_image": 0,
             "skipped_empty_function": 0
         }
        
        samples_list = list(data_generator)
        print(f"Loaded {len(samples_list)} samples")
        iterator = tqdm(samples_list, desc="Processing samples", unit="sample") if TQDM_AVAILABLE else samples_list
        
        for sample in iterator:
            if "uia_controls_info" not in sample["control_infos"]:
                continue
            domain = sample["domain"]
            sample_id = sample["sample_id"]
            
            # Track domain statistics
            if domain not in domain_stats:
                domain_stats[domain] = 0
            domain_stats[domain] += 1
            
            # Get domain-specific supported actions
            supported_actions = self.get_supported_actions_a11y(domain)
            
            # Format history of actions
            if sample["previous_actions"]:
                history = "\n".join(sample["previous_actions"])
            else:
                history = ""

            # Transform a11y information (will be scaled later if image is resized)
            # transformed_a11y = self.transform_a11y(sample["control_infos"])
            transformed_a11y = self.clean_a11y(sample["control_infos"])
            
            # Format the action as a tool call
            action_data = sample["action"]
            try:
                # Parse the action if it's a string
                if isinstance(action_data, str):
                    action_data = json.loads(action_data)
                
                # Extract function name and arguments from action data
                function_name = ""
                args = {}
                
                if "function" in action_data:
                    function_name = action_data["function"]
                    # Copy all other fields as arguments, excluding 'action'
                    args = action_data["args"]
                
                # Filter out actions with empty function name and status not OVERALL_FINISH
                if not function_name and sample["status"] != "FINISH":
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Skipping sample {sample_id}: empty function and status is not FINISH")
                    else:
                        print(f"Skipping sample {sample_id}: empty function and status is not FINISH")
                    processing_stats["skipped_empty_function"] = processing_stats.get("skipped_empty_function", 0) + 1
                    continue

                if action_data["function"] == "drag":
                    args["start_coordinate"] = [args["start_x"], args["start_y"]]
                    args["end_coordinate"] = [args["end_x"], args["end_y"]]

                    del args["start_x"]
                    del args["start_y"]
                    del args["end_x"]
                    del args["end_y"]
                else:
                    if action_data["control_label"]:
                        args["control_label"] = action_data["control_label"]
                    else:
                        args["coordinate"] = [action_data["coordinate_x"], action_data["coordinate_y"]]
                    if "x" in args:
                        del args["x"]
                    if "y" in args:
                        del args["y"]
                    
                # Create the tool call format
                tool_call = {
                    "function": function_name,
                    "args": args,
                    "status": sample["status"]
                }
                
                # Convert tool call to JSON string for coordinate processing
                tool_call_json = json.dumps(tool_call, indent=4)
                
                # Format the GPT response with thought and tool call
                thought_text = sample["thought"]
                gpt_response = f"<tool_call>\n{tool_call_json}\n</tool_call>"
                 
            except Exception as e:
                 if TQDM_AVAILABLE:
                    tqdm.write(f"Error processing action for sample {sample_id}")
                 else:
                    import traceback
                    print(f"Error processing action for sample {sample_id}: {e}")
                    traceback.print_exc()
                 continue
            
            # Process image and coordinates
            original_image_path = sample["screenshot_annotated"]
            
            if not os.path.exists(original_image_path):
                if TQDM_AVAILABLE:
                    tqdm.write(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                else:
                    print(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                processing_stats["skipped_no_image"] += 1
                continue
            
            # Check if the response contains coordinates that need scaling
            has_coordinates = any(coord_type in tool_call_json for coord_type in ["coordinate", "start_coordinate", "end_coordinate"])
            bbox = None
            if not has_coordinates:
                # No coordinates to process, just copy image and save sample
                try:
                    with Image.open(original_image_path) as img:
                        original_width, original_height = img.size
                        
                        # Calculate new dimensions using smart_resize
                        if QWEN_VL_UTILS_AVAILABLE:
                            new_height, new_width = smart_resize(
                                height=original_height,
                                width=original_width,
                                factor=factor,
                                max_pixels=max_pixels
                            )
                        else:
                            new_height, new_width = self.smart_resize_fallback(
                                height=original_height,
                                width=original_width,
                                factor=factor,
                                max_pixels=max_pixels
                            )
                        
                        # Calculate scaling factors
                        scale_w = new_width / original_width
                        scale_h = new_height / original_height
                        
                                                 # Resize image and save
                        resized_img = img.resize((new_width, new_height))
                        
                        # Extract jsonl filename and create images directory
                        parts = sample_id.split('_')
                        if len(parts) >= 4:
                            jsonl_filename = '_'.join(parts[-4:-1])
                        else:
                            jsonl_filename = "unknown"
                        
                        images_dir = os.path.join(output_dir, "images", jsonl_filename)
                        os.makedirs(images_dir, exist_ok=True)
                        
                        clean_filename = os.path.basename(original_image_path)
                        clean_dest = os.path.join(images_dir, clean_filename)
                        resized_img.save(clean_dest)
                        
                        image_paths = [f"images/{jsonl_filename}/{clean_filename}"]

                        processing_stats["images_resized"] += 1
                        
                        # Scale a11y rectangles to match the resized image
                        # final_a11y = self.scale_a11y_rectangles(transformed_a11y, scale_w, scale_h)
                    
                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Error copying image for {sample_id}: {e}")
                    else:
                        print(f"Error copying image for {sample_id}: {e}")
                    continue
            else:
                # Process image resizing and coordinate scaling
                try:
                    with Image.open(original_image_path) as img:
                        original_width, original_height = img.size
                        
                        # Calculate new dimensions using smart_resize
                        if QWEN_VL_UTILS_AVAILABLE:
                            new_height, new_width = smart_resize(
                                height=original_height,
                                width=original_width,
                                factor=factor,
                                max_pixels=max_pixels
                            )
                        else:
                            new_height, new_width = self.smart_resize_fallback(
                                height=original_height,
                                width=original_width,
                                factor=factor,
                                max_pixels=max_pixels
                            )
                        
                        # Calculate scaling factors
                        scale_w = new_width / original_width
                        scale_h = new_height / original_height
                        
                                                 # Resize image and save
                        resized_img = img.resize((new_width, new_height))
                        
                        # Extract jsonl filename and create images directory
                        parts = sample_id.split('_')
                        if len(parts) >= 4:
                            jsonl_filename = '_'.join(parts[-4:-1])
                        else:
                            jsonl_filename = "unknown"
                        
                        images_dir = os.path.join(output_dir, "images", jsonl_filename)
                        os.makedirs(images_dir, exist_ok=True)
                        
                        clean_filename = os.path.basename(original_image_path)
                        clean_dest = os.path.join(images_dir, clean_filename)
                        resized_img.save(clean_dest)
                        
                        image_paths = [f"images/{jsonl_filename}/{clean_filename}"]
                        
                        # Scale coordinates in the GPT response
                        updated_gpt_response = self.extract_and_scale_coordinates(gpt_response, scale_w, scale_h)
                        gpt_response = updated_gpt_response
                        
                        # Scale a11y rectangles to match the resized image
                        # final_a11y = self.scale_a11y_rectangles(transformed_a11y, scale_w, scale_h)

                        if action_data["rectangle"]:
                            bbox = [
                                action_data["rectangle"]["left"] * scale_w,
                                action_data["rectangle"]["top"] * scale_h,
                                action_data["rectangle"]["right"] * scale_w,
                                action_data["rectangle"]["bottom"] * scale_h
                            ]
                        else:
                            bbox = None
                        
                        processing_stats["images_resized"] += 1
                        processing_stats["coordinates_scaled"] += 1
                        
                        if TQDM_AVAILABLE:
                            tqdm.write(f"✅ Processed {sample_id}: {original_width}x{original_height} → {new_width}x{new_height}")
                        
                except Exception as e:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"❌ Error processing image for {sample_id}: {e}")
                    else:
                        print(f"❌ Error processing image for {sample_id}: {e}")
                    continue
            
            # Create the human message with the final (possibly scaled) a11y information
            human_message = ACTION_PREDICTION_USER_PROMPT_QWEN_A11Y.format(
                instruction=sample["request"],
                a11y=transformed_a11y["uia_controls_info"],
                history=history,
                actions=supported_actions
            )
            
            # Create the training sample
            training_sample = {
                "id": sample_id,
                "images": image_paths,
                "conversation": [
                    {
                        "from": "human",
                        "value": f"<image>\n{human_message}"
                    },
                    {
                        "from": "gpt", 
                        "value": gpt_response
                    }
                ],
                "reward": 1 if success_or_fail == "success" else 0,
                "bbox": bbox,
                "label": action_data["control_label"] if "control_label" in action_data else None
            }
            
            training_data.append(training_sample)
            processing_stats["total_processed"] += 1
        
            # Print statistics
        print(f"\n=== Processing Statistics ===")
        print(f"Total samples processed: {processing_stats['total_processed']}")
        print(f"Images resized: {processing_stats['images_resized']}")
        print(f"Coordinates scaled: {processing_stats['coordinates_scaled']}")
        print(f"Skipped (no coordinates): {processing_stats['skipped_no_coordinates']}")
        print(f"Skipped (no image): {processing_stats['skipped_no_image']}")
        print(f"Skipped (empty function): {processing_stats['skipped_empty_function']}")
        
        print(f"\n=== Domain Statistics ===")
        for domain, count in domain_stats.items():
            print(f"{domain}: {count} samples")
        
        return training_data
    
    def convert_to_screen_parsing_resize(self, root_dir, success_or_fail, output_dir, max_pixels=999999, factor=28):
        """
        Convert data to action prediction training format with image resizing and coordinate scaling.
        
        Args:
            root_dir (str): Root directory of the dataset
            success_or_fail (str): Which subset to process ('success' or 'fail')
            output_dir (str): Output directory for processed data and images
            max_pixels (int): Maximum number of pixels for resized images
            factor (int): Factor that image dimensions must be divisible by
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for image processing. Please install it: pip install Pillow")
        
        data_generator = self.load_data(
            task_type="screen_parsing", 
            root_dir=root_dir, 
            success_or_fail=success_or_fail
        )
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        processed_images_dir = os.path.join(output_dir, "images")
        os.makedirs(processed_images_dir, exist_ok=True)
        
        print(f"🖼️  Resized images will be saved in: {processed_images_dir}")
        print(f"📄 Training data will be saved in: {output_dir}")
        
        training_data = []
        domain_stats = {}
        processing_stats = {
             "total_processed": 0,
             "images_resized": 0,
             "coordinates_scaled": 0,
             "skipped_no_coordinates": 0,
             "skipped_no_image": 0,
             "skipped_empty_function": 0
         }
        
        samples_list = list(data_generator)
        print(f"Loaded {len(samples_list)} samples")
        iterator = tqdm(samples_list, desc="Processing samples", unit="sample") if TQDM_AVAILABLE else samples_list
        
        for sample in iterator:
            domain = sample["domain"]
            sample_id = sample["sample_id"]

            control_infos = sample["control_infos"]
            
            if control_infos.get("uia_controls_info", None) is None:
                continue

            control_infos = self.transform_a11y(control_infos)
            control_infos = control_infos.get("uia_controls_info", [])

            # Create the human message with the prompt using domain-specific actions
            human_message = SCREEN_PARSING_SYS_PROMPT
            
            
            
            # Process image and coordinates
            original_image_path = sample["screenshot_clean"]
            
            if not os.path.exists(original_image_path):
                if TQDM_AVAILABLE:
                    tqdm.write(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                else:
                    print(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                processing_stats["skipped_no_image"] += 1
                continue

            
            # Process image resizing and coordinate scaling
            try:
                with Image.open(original_image_path) as img:
                    original_width, original_height = img.size
                    
                    # Calculate new dimensions using smart_resize
                    if QWEN_VL_UTILS_AVAILABLE:
                        new_height, new_width = smart_resize(
                            height=original_height,
                            width=original_width,
                            factor=factor,
                            max_pixels=max_pixels
                        )
                    else:
                        new_height, new_width = self.smart_resize_fallback(
                            height=original_height,
                            width=original_width,
                            factor=factor,
                            max_pixels=max_pixels
                        )

                    scale_w = new_width / original_width
                    scale_h = new_height / original_height

                    for control_info in control_infos:
                        control_info["control_rect"] = [
                            control_info["control_rect"][0] * scale_w,
                            control_info["control_rect"][1] * scale_h,
                            control_info["control_rect"][2] * scale_w,
                            control_info["control_rect"][3] * scale_h
                        ]
                    
                                                # Resize image and save
                    resized_img = img.resize((new_width, new_height))
                    
                    # Extract jsonl filename and create images directory
                    parts = sample_id.split('_')
                    if len(parts) >= 4:
                        jsonl_filename = '_'.join(parts[-4:-1])
                    else:
                        jsonl_filename = "unknown"
                    
                    images_dir = os.path.join(output_dir, "images", jsonl_filename)
                    os.makedirs(images_dir, exist_ok=True)
                    
                    clean_filename = os.path.basename(original_image_path)
                    clean_dest = os.path.join(images_dir, clean_filename)
                    resized_img.save(clean_dest)
                    
                    image_paths = [f"images/{jsonl_filename}/{clean_filename}"]
                    
                    processing_stats["images_resized"] += 1
                    processing_stats["coordinates_scaled"] += 1
                    
                    if TQDM_AVAILABLE:
                        tqdm.write(f"✅ Processed {sample_id}: {original_width}x{original_height} → {new_width}x{new_height}")
                    
            except Exception as e:
                if TQDM_AVAILABLE:
                    tqdm.write(f"❌ Error processing image for {sample_id}: {e}")
                else:
                    print(f"❌ Error processing image for {sample_id}: {e}")
                continue
            
            # Create the training sample
            training_sample = {
                "id": sample_id,
                "images": image_paths,
                "conversation": [
                    {
                        "from": "human",
                        "value": f"<image>\n{human_message}"
                    },
                    {
                        "from": "gpt", 
                        "value": json.dumps(control_infos)
                    }
                ]
            }
            
            training_data.append(training_sample)
            processing_stats["total_processed"] += 1
        
            # Print statistics
        print(f"\n=== Processing Statistics ===")
        print(f"Total samples processed: {processing_stats['total_processed']}")
        print(f"Images resized: {processing_stats['images_resized']}")
        print(f"Coordinates scaled: {processing_stats['coordinates_scaled']}")
        print(f"Skipped (no coordinates): {processing_stats['skipped_no_coordinates']}")
        print(f"Skipped (no image): {processing_stats['skipped_no_image']}")
        print(f"Skipped (empty function): {processing_stats['skipped_empty_function']}")
        
        print(f"\n=== Domain Statistics ===")
        for domain, count in domain_stats.items():
            print(f"{domain}: {count} samples")
        
        return training_data

    def convert_to_grounding_resize(self, root_dir, success_or_fail, output_dir, max_pixels=999999, factor=28):
        """
        Convert data to grounding training format with image resizing and coordinate scaling.
        
        Args:
            root_dir (str): Root directory of the dataset
            success_or_fail (str): Which subset to process ('success' or 'fail')
            output_dir (str): Output directory for processed data and images
            max_pixels (int): Maximum number of pixels for resized images
            factor (int): Factor that image dimensions must be divisible by
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for image processing. Please install it: pip install Pillow")
        
        data_generator = self.load_data(
            task_type="action_prediction", 
            root_dir=root_dir, 
            success_or_fail=success_or_fail
        )
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        processed_images_dir = os.path.join(output_dir, "images")
        os.makedirs(processed_images_dir, exist_ok=True)
        
        print(f"🖼️  Resized images will be saved in: {processed_images_dir}")
        print(f"📄 Training data will be saved in: {output_dir}")
        
        training_data = []
        domain_stats = {}
        processing_stats = {
             "total_processed": 0,
             "images_resized": 0,
             "coordinates_scaled": 0,
             "skipped_no_coordinates": 0,
             "skipped_no_image": 0,
             "skipped_empty_function": 0
         }
        
        samples_list = list(data_generator)
        print(f"Loaded {len(samples_list)} samples")
        iterator = tqdm(samples_list, desc="Processing grounding samples", unit="sample") if TQDM_AVAILABLE else samples_list
        
        for sample in iterator:
            domain = sample["domain"]
            sample_id = sample["sample_id"]
            
            # Track domain statistics
            if domain not in domain_stats:
                domain_stats[domain] = 0
            domain_stats[domain] += 1
            
            # Parse action data to get coordinates
            action_data = sample["action"]
            try:
                # Parse the action if it's a string
                if isinstance(action_data, str):
                    action_data = json.loads(action_data)
                
                # Extract coordinates from action data
                coordinate_x = None
                coordinate_y = None
                
                if action_data.get("function") == "drag":
                    # For drag actions, use start coordinates
                    args = action_data.get("args", {})
                    coordinate_x = args.get("start_x")
                    coordinate_y = args.get("start_y")
                else:
                    # For other actions, use coordinate_x and coordinate_y
                    coordinate_x = action_data.get("coordinate_x")
                    coordinate_y = action_data.get("coordinate_y")
                
                # Skip samples without valid coordinates
                if coordinate_x is None or coordinate_y is None:
                    if TQDM_AVAILABLE:
                        tqdm.write(f"Skipping sample {sample_id}: no valid coordinates")
                    else:
                        print(f"Skipping sample {sample_id}: no valid coordinates")
                    processing_stats["skipped_no_coordinates"] += 1
                    continue
                    
            except Exception as e:
                if TQDM_AVAILABLE:
                    tqdm.write(f"Error processing action for sample {sample_id}: {e}")
                else:
                    print(f"Error processing action for sample {sample_id}: {e}")
                continue
            
            # Process image and coordinates
            original_image_path = sample["screenshot_clean"]
            
            if not os.path.exists(original_image_path):
                if TQDM_AVAILABLE:
                    tqdm.write(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                else:
                    print(f"⚠️ Warning: Skipping {sample_id} - image not found: {original_image_path}")
                processing_stats["skipped_no_image"] += 1
                continue
            
            # Process image resizing and coordinate scaling
            try:
                with Image.open(original_image_path) as img:
                    original_width, original_height = img.size
                    
                    # Calculate new dimensions using smart_resize
                    if QWEN_VL_UTILS_AVAILABLE:
                        new_height, new_width = smart_resize(
                            height=original_height,
                            width=original_width,
                            factor=factor,
                            max_pixels=max_pixels
                        )
                    else:
                        new_height, new_width = self.smart_resize_fallback(
                            height=original_height,
                            width=original_width,
                            factor=factor,
                            max_pixels=max_pixels
                        )
                    
                    # Calculate scaling factors
                    scale_w = new_width / original_width
                    scale_h = new_height / original_height
                    
                    # Scale coordinates
                    scaled_x = coordinate_x * scale_w
                    scaled_y = coordinate_y * scale_h
                    
                    # Resize image and save
                    resized_img = img.resize((new_width, new_height))
                    
                    # Extract jsonl filename and create images directory
                    parts = sample_id.split('_')
                    if len(parts) >= 4:
                        jsonl_filename = '_'.join(parts[-4:-1])
                    else:
                        jsonl_filename = "unknown"
                    
                    images_dir = os.path.join(output_dir, "images", jsonl_filename)
                    os.makedirs(images_dir, exist_ok=True)
                    
                    clean_filename = os.path.basename(original_image_path)
                    clean_dest = os.path.join(images_dir, clean_filename)
                    resized_img.save(clean_dest)
                    
                    image_paths = [f"images/{jsonl_filename}/{clean_filename}"]
                    
                    processing_stats["images_resized"] += 1
                    processing_stats["coordinates_scaled"] += 1
                    
                    if TQDM_AVAILABLE:
                        tqdm.write(f"✅ Processed {sample_id}: {original_width}x{original_height} → {new_width}x{new_height}")
                    
            except Exception as e:
                if TQDM_AVAILABLE:
                    tqdm.write(f"❌ Error processing image for {sample_id}: {e}")
                else:
                    print(f"❌ Error processing image for {sample_id}: {e}")
                continue
            
            # Create the human message using grounding prompt
            human_message = GROUNDING_USER_PROMPT_QWEN.format(instruction=sample["thought"])
            
            # Create the GPT response with scaled coordinates
            gpt_response = f"<coordinate> [{scaled_x:.0f}, {scaled_y:.0f}] </coordinate>"
            
            # Create the training sample
            training_sample = {
                "id": sample_id,
                "images": image_paths,
                "conversation": [
                    {
                        "from": "human",
                        "value": f"<image>\n{human_message}"
                    },
                    {
                        "from": "gpt", 
                        "value": gpt_response
                    }
                ]
            }
            
            training_data.append(training_sample)
            processing_stats["total_processed"] += 1
        
        # Print statistics
        print(f"\n=== Processing Statistics ===")
        print(f"Total samples processed: {processing_stats['total_processed']}")
        print(f"Images resized: {processing_stats['images_resized']}")
        print(f"Coordinates scaled: {processing_stats['coordinates_scaled']}")
        print(f"Skipped (no coordinates): {processing_stats['skipped_no_coordinates']}")
        print(f"Skipped (no image): {processing_stats['skipped_no_image']}")
        print(f"Skipped (empty function): {processing_stats['skipped_empty_function']}")
        
        print(f"\n=== Domain Statistics ===")
        for domain, count in domain_stats.items():
            print(f"{domain}: {count} samples")
        
        return training_data

    def save_training_data(self, training_data, output_file):
        """Save training data to a JSON file."""
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {len(training_data)} samples to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(training_data)} training samples to {output_file}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Convert GUI action data to training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic action prediction conversion
  python convert_to_train.py --root_dir F:/dataset --output_dir F:/output

  # Convert to grounding format with image resizing
  python convert_to_train.py --root_dir F:/dataset --output_dir F:/output --type grounding --resize --max_pixels 500000

  # Convert to action prediction with a11y information
  python convert_to_train.py --root_dir F:/dataset --output_dir F:/output --type action_prediction_a11y --resize

  # Convert to screen parsing format
  python convert_to_train.py --root_dir F:/dataset --output_dir F:/output --type screen_parsing --resize
  
  # Process only failed samples for grounding
  python convert_to_train.py --root_dir F:/dataset --output_dir F:/output --type grounding --success_or_fail fail
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--root_dir", 
        type=str, 
        required=True,
        help="Root directory of the dataset (contains 'data' and 'image' folders)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for processed data and images"
    )
    
    # Optional arguments
    parser.add_argument(
        "--success_or_fail", 
        type=str, 
        choices=["success", "fail"], 
        default="success",
        help="Process 'success' or 'fail' samples (default: success)"
    )
    
    parser.add_argument(
        "--type", 
        type=str,
        choices=["action_prediction", "action_prediction_a11y", "screen_parsing", "grounding"],
        default="action_prediction",
        help="Type of data conversion (default: action_prediction)"
    )

    parser.add_argument(
        "--resize", 
        action="store_true",
        help="Enable image resizing and coordinate scaling"
    )
    
    parser.add_argument(
        "--max_pixels", 
        type=int, 
        default=999999,
        help="Maximum number of pixels for resized images (default: 999999)"
    )
    
    parser.add_argument(
        "--factor", 
        type=int, 
        default=28,
        help="Factor that image dimensions must be divisible by (default: 28)"
    )
    
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default=None,
        help="Output JSON filename (default: auto-generated based on conversion type)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("🚀 GUI Action Data Converter")
    print("=" * 50)
    print(f"📥 Input directory: {args.root_dir}")
    print(f"📤 Output directory: {args.output_dir}")
    print(f"🎯 Processing: {args.success_or_fail} samples")
    print(f"📋 Conversion type: {args.type}")
    print(f"🖼️  Image resizing: {'Enabled' if args.resize else 'Disabled'}")
    if args.resize:
        print(f"   Max pixels: {args.max_pixels:,}")
        print(f"   Alignment factor: {args.factor}")
    
    print(f"📄 Output filename: {args.output_filename or 'auto-generated'}")
    print("-" * 50)
    
    # Validate input directory
    if not os.path.exists(args.root_dir):
        print(f"❌ Error: Root directory not found: {args.root_dir}")
        return 1
    
    data_dir = os.path.join(args.root_dir, "data")
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        return 1
    
    try:
        convertor = Convertor()

        if args.type == "grounding":
            print("🔄 Starting conversion with grounding...")
            training_data = convertor.convert_to_grounding_resize(
                root_dir=args.root_dir,
                success_or_fail=args.success_or_fail,
                output_dir=args.output_dir,
                max_pixels=args.max_pixels,
                factor=args.factor
            )
        elif args.type == "screen_parsing":
            print("🔄 Starting conversion with screen parsing...")
            training_data = convertor.convert_to_screen_parsing_resize(
                root_dir=args.root_dir,
                success_or_fail=args.success_or_fail,
                output_dir=args.output_dir,
                max_pixels=args.max_pixels,
                factor=args.factor
            )
        elif args.type == "action_prediction_a11y":
            print("🔄 Starting conversion with a11y information...")
            training_data = convertor.convert_to_action_prediction_resize_a11y(
                root_dir=args.root_dir,
                success_or_fail=args.success_or_fail,
                output_dir=args.output_dir,
                max_pixels=args.max_pixels,
                factor=args.factor
            )
        elif args.type == "action_prediction":
            # Choose conversion method based on resize flag
            if args.resize:
                print("🔄 Starting conversion with image resizing...")
                training_data = convertor.convert_to_action_prediction_resize(
                    root_dir=args.root_dir,
                    success_or_fail=args.success_or_fail,
                    output_dir=args.output_dir,
                    max_pixels=args.max_pixels,
                    factor=args.factor
                )
            else:
                print("🔄 Starting basic conversion...")
                training_data = convertor.convert_to_action_prediction(
                    root_dir=args.root_dir,
                    success_or_fail=args.success_or_fail,
                    output_dir=args.output_dir
                )
        else:
            raise ValueError(f"Unsupported conversion type: {args.type}")
        
        # Generate output filename based on conversion type
        if args.output_filename is None:
            if args.type == "grounding":
                default_filename = "grounding_training_data.json"
            elif args.type == "screen_parsing":
                default_filename = "screen_parsing_training_data.json"
            elif args.type == "action_prediction_a11y":
                default_filename = "action_prediction_a11y_training_data.json"
            else:  # action_prediction
                default_filename = "action_prediction_training_data.json"
            output_filename = default_filename
        else:
            output_filename = args.output_filename
        
        # Save training data
        output_file = os.path.join(args.output_dir, output_filename)
        convertor.save_training_data(training_data, output_file)
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📊 Generated {len(training_data)} training samples")
        print(f"💾 Data saved to: {output_file}")
        
        if args.verbose and training_data:
            print(f"\n=== Sample Preview ===")
            sample = training_data[0]
            print(f"ID: {sample['id']}")
            print(f"Images: {sample['images']}")
            print(f"Conversation length: {len(sample['conversation'])}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())