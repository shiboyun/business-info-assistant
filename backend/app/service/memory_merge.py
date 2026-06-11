"""记忆合并策略模块"""
from schemas.memory_config import MergeOp


def apply_merge(current_value: str | None, patch_value: str, merge_op: MergeOp) -> str:
    """对字段值应用合并策略。

    Args:
        current_value: 当前字段值（None 表示新字段）
        patch_value: LLM 输出的新值
        merge_op: 合并策略

    Returns:
        合并后的字段值
    """
    # IMMUTABLE：已有值不允许修改, 仅首次可写入
    if merge_op == MergeOp.IMMUTABLE and current_value is not None:
        return current_value

    # REPLACE 或新字段（任何策略下无现有内容可合并）：直接设置
    if merge_op == MergeOp.REPLACE or current_value is None:
        return patch_value

    if merge_op == MergeOp.SUM:
        return current_value + "\n" + patch_value if current_value else patch_value

    if merge_op == MergeOp.PATCH:
        # SEARCH/REPLACE 文本替换
        # LLM 输出格式: "SEARCH: <旧文本>\nREPLACE: <新文本>"
        # 多个块用 "---" 分隔
        blocks = patch_value.split("---")
        result = current_value
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if "SEARCH:" in block and "REPLACE:" in block:
                parts = block.split("REPLACE:", 1)
                search = parts[0].replace("SEARCH:", "").strip()
                replace = parts[1].strip()
                if search in result:
                    result = result.replace(search, replace)  ##兜底策略完善：存在多个匹配时，直接对整段替换
                else:
                    # search 不匹配时追加到末尾
                    result = result + "\n" + replace
            else:
                # 非标准格式，直接追加
                if result and block not in result:
                    result = result + "\n" + block
        return result

    return patch_value
