import re
import json
from typing import Optional, Dict, List, Union, Tuple
from .tool_definitions import normalize_tool_args

def extract_json_from_tool_call(text: str) -> Optional[Union[Dict, List]]:
    """
    Extracts and parses a JSON object from text containing <tool_call> tags using regular expressions.

    Args:
        text: The raw string containing the <tool_call>...</tool_call> format.

    Returns:
        The parsed JSON object (usually a dict or list) if extraction and parsing are successful.
        None if no matching tags are found, or if the content within the tags is not valid JSON.
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        json_string = match.group(1).strip()
        
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Warning: Content found within <tool_call> tags, but it is not valid JSON. Content: '{json_string[:100]}...'")
            return None
            
    return None


def eval_tool(predict: dict | str, ground_truth: dict | str, ground_bbox: list | dict | None = None) -> Tuple[int, int, int]:
    """
    Evaluate the tool prediction against the ground truth.

    Args:
        predict: The predicted tool call.
        ground_truth: The ground truth tool call.
        ground_bbox: The ground truth bounding box.

    Returns:
        A tuple of (function_match, args_match, status_match). For example: (1, 1, 1)
    """
    if isinstance(predict, str):
        predict = extract_json_from_tool_call(predict)
    if isinstance(ground_truth, str):
        ground_truth = extract_json_from_tool_call(ground_truth)

    def _to_rect(rect_like):
        if rect_like is None:
            return None
        if isinstance(rect_like, dict):
            keys = {k.lower() for k in rect_like.keys()}
            if {"left", "top", "right", "bottom"}.issubset(keys):
                return {
                    "left": float(rect_like["left"]),
                    "top": float(rect_like["top"]),
                    "right": float(rect_like["right"]),
                    "bottom": float(rect_like["bottom"]),
                }
        if isinstance(rect_like, (list, tuple)) and len(rect_like) == 4:
            l, t, r, b = rect_like
            try:
                return {
                    "left": float(l),
                    "top": float(t),
                    "right": float(r),
                    "bottom": float(b),
                }
            except (TypeError, ValueError):
                return None
        return None

    def _split_rects(bbox: Union[List, Dict, None]) -> Tuple[Optional[Dict], Optional[Dict]]:
        if bbox is None:
            return None, None
        if isinstance(bbox, dict):
            return _to_rect(bbox), None
        if isinstance(bbox, list):
            # Support [rect] or [start_rect, end_rect] or [l,t,r,b]
            if len(bbox) == 1:
                return _to_rect(bbox[0]), None
            if len(bbox) >= 2:
                return _to_rect(bbox[0]), _to_rect(bbox[1])
            if len(bbox) == 4:
                return _to_rect(bbox), None
        return None, None

    def _extract_action(obj: Union[Dict, List, None]) -> Tuple[Optional[str], Dict, Optional[str]]:
        if not obj:
            return None, {}, None
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if not isinstance(obj, dict):
            return None, {}, None

        # Common fields
        func = obj.get("function") or obj.get("name") or obj.get("tool")
        if not func and "action" in obj and isinstance(obj["action"], dict):
            func = obj["action"].get("function")

        args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
        if not args and "action" in obj and isinstance(obj["action"], dict):
            args = obj["action"].get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        status = obj.get("status")
        if status is None and "action" in obj and isinstance(obj["action"], dict):
            status = obj["action"].get("status")

        # Normalize coordinates like in action_prediction.py
        if isinstance(args, dict):
            if func == "drag":
                if {"start_x", "start_y", "end_x", "end_y"}.issubset(args.keys()):
                    sx, sy, ex, ey = args.get("start_x"), args.get("start_y"), args.get("end_x"), args.get("end_y")
                    try:
                        args["start_coordinate"] = [float(sx), float(sy)]
                        args["end_coordinate"] = [float(ex), float(ey)]
                        for k in ["start_x", "start_y", "end_x", "end_y"]:
                            args.pop(k, None)
                    except (TypeError, ValueError):
                        pass
            else:
                if {"x", "y"}.issubset(args.keys()):
                    try:
                        args["coordinate"] = [float(args.get("x")), float(args.get("y"))]
                        args.pop("x", None)
                        args.pop("y", None)
                    except (TypeError, ValueError):
                        pass
                if {"coordinate_x", "coordinate_y"}.issubset(obj.keys()):
                    try:
                        args["coordinate"] = [float(obj.get("coordinate_x")), float(obj.get("coordinate_y"))]
                    except (TypeError, ValueError):
                        pass

        return (str(func).lower() if func else None), (args if isinstance(args, dict) else {}), (str(status) if status is not None else None)

    if predict is None or ground_truth is None:
        return 0, 0, 0

    pred_func, pred_args, pred_status = _extract_action(predict)
    gt_func, gt_args, gt_status = _extract_action(ground_truth)

    # Function match
    function_match = pred_func == gt_func if (pred_func and gt_func) else False

    # Status match (normalize common aliases)
    def _norm_status(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s_norm = str(s).strip().upper()
        if s_norm == "OVERALL_FINISH":
            return "FINISH"
        return s_norm

    status_match = _norm_status(pred_status) == _norm_status(gt_status) if (pred_status is not None and gt_status is not None) else False

    # Args match with rectangle/tolerance logic
    def _compare_drag_args(p_args: Dict, g_args: Dict, rect_start: Optional[Dict], rect_end: Optional[Dict]) -> bool:
        try:
            p_norm = normalize_tool_args("drag", p_args)
            g_norm = normalize_tool_args("drag", g_args)

            if "start_coordinate" not in p_norm or "end_coordinate" not in p_norm:
                return False
            if "start_coordinate" not in g_norm or "end_coordinate" not in g_norm:
                return False

            ps = p_norm["start_coordinate"]
            pe = p_norm["end_coordinate"]
            gs = g_norm["start_coordinate"]
            ge = g_norm["end_coordinate"]

            if not (isinstance(ps, (list, tuple)) and len(ps) == 2):
                return False
            if not (isinstance(pe, (list, tuple)) and len(pe) == 2):
                return False
            if not (isinstance(gs, (list, tuple)) and len(gs) == 2):
                return False
            if not (isinstance(ge, (list, tuple)) and len(ge) == 2):
                return False

            tol = 25.0

            def _in_rect(coord, rect):
                if not rect:
                    return None
                x, y = float(coord[0]), float(coord[1])
                return rect["left"] <= x <= rect["right"] and rect["top"] <= y <= rect["bottom"]

            start_match = _in_rect(ps, rect_start)
            if start_match is None:
                start_match = abs(float(ps[0]) - float(gs[0])) <= tol and abs(float(ps[1]) - float(gs[1])) <= tol

            end_match = _in_rect(pe, rect_end)
            if end_match is None:
                end_match = abs(float(pe[0]) - float(ge[0])) <= tol and abs(float(pe[1]) - float(ge[1])) <= tol

            other_ok = True
            for key in ["button", "duration", "key_hold"]:
                pv = p_norm.get(key)
                gv = g_norm.get(key)
                pv_str = str(pv).lower() if pv is not None else "none"
                gv_str = str(gv).lower() if gv is not None else "none"
                if pv_str != gv_str:
                    other_ok = False
                    break

            return bool(start_match and end_match and other_ok)
        except Exception:
            return False

    def _compare_regular_args(p_args: Dict, g_args: Dict, rect: Optional[Dict], p_func: str, g_func: str) -> bool:
        try:
            p_norm = normalize_tool_args(p_func, p_args)
            g_norm = normalize_tool_args(g_func, g_args)

            if "coordinate" in p_norm and "coordinate" in g_norm:
                pc = p_norm["coordinate"]
                gc = g_norm["coordinate"]
                if (
                    isinstance(pc, (list, tuple)) and len(pc) == 2 and isinstance(gc, (list, tuple)) and len(gc) == 2
                ):
                    tol = 25.0
                    if rect:
                        x, y = float(pc[0]), float(pc[1])
                        coord_ok = rect["left"] <= x <= rect["right"] and rect["top"] <= y <= rect["bottom"]
                    else:
                        coord_ok = abs(float(pc[0]) - float(gc[0])) <= tol and abs(float(pc[1]) - float(gc[1])) <= tol

                    other_ok = True
                    for key in p_norm:
                        if key == "coordinate":
                            continue
                        pv = p_norm.get(key)
                        gv = g_norm.get(key)
                        pv_str = str(pv).lower() if pv is not None else "none"
                        gv_str = str(gv).lower() if gv is not None else "none"
                        if pv_str != gv_str:
                            other_ok = False
                            break

                    return bool(coord_ok and other_ok)

            # Fallback: compare normalized dicts with string-normalization for scalars
            def _to_cmp(d: Dict) -> Dict:
                out = {}
                for k, v in d.items():
                    if isinstance(v, (str, bool)):
                        out[k] = str(v).lower()
                    elif v is None:
                        out[k] = "none"
                    else:
                        out[k] = v
                return out

            return _to_cmp(p_norm) == _to_cmp(g_norm)
        except Exception:
            return False

    rect_start, rect_end = _split_rects(ground_bbox)

    if pred_func and gt_func and pred_args is not None and gt_args is not None:
        if pred_func == "drag" and gt_func == "drag":
            args_match = _compare_drag_args(pred_args, gt_args, rect_start, rect_end)
        else:
            args_match = _compare_regular_args(pred_args, gt_args, rect_start, pred_func or "unknown", gt_func or "unknown")
    else:
        args_match = False

    return int(function_match), int(args_match), int(status_match)
