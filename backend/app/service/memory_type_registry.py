"""记忆类型注册表 - 加载 YAML 配置"""
import os
from pathlib import Path
from typing import Optional

import yaml

from schemas.memory_config import MemoryTypeSchema, MemoryField, MergeOp, FieldType


class MemoryTypeRegistry:
    """记忆类型注册表，从 YAML 文件加载 schema 配置。"""

    def __init__(self, config_dir: str | None = None):
        self._types: dict[str, MemoryTypeSchema] = {}
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "memory_types"
            )
        self._config_dir = config_dir
        self._load_all()

    def _load_all(self) -> None:
        """扫描配置目录，加载所有 .yaml 文件。"""
        dir_path = Path(self._config_dir)
        if not dir_path.exists():
            print(f"[MemoryTypeRegistry] 配置目录不存在: {self._config_dir}")
            return

        for yaml_file in sorted(dir_path.glob("*.yaml")):
            try:
                self._load_file(str(yaml_file))
                print(f"[MemoryTypeRegistry] 已加载: {yaml_file.name}")
            except Exception as e:
                print(f"[MemoryTypeRegistry] 加载失败 {yaml_file.name}: {e}")

    def _load_file(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        fields = []
        for fd in data.get("fields", []):
            fields.append(MemoryField(
                name=fd["name"],
                type=FieldType(fd.get("type", "string")),
                description=fd.get("description", ""),
                merge_op=MergeOp(fd.get("merge_op", "patch")),
            ))

        schema = MemoryTypeSchema(
            memory_type=data["memory_type"],
            description=data.get("description", ""),
            directory=data.get("directory", ""),
            filename_template=data.get("filename_template", ""),
            embedding_template=data.get("embedding_template", ""),
            fields=fields,
        )
        self._types[schema.memory_type] = schema

    def get(self, name: str) -> Optional[MemoryTypeSchema]:
        """按名称获取 schema。"""
        return self._types.get(name)

    def list_all(self) -> list[MemoryTypeSchema]:
        """获取所有已注册的记忆类型。"""
        return list(self._types.values())

    def list_names(self) -> list[str]:
        """获取所有记忆类型名称。"""
        return list(self._types.keys())


# 单例
_registry: Optional[MemoryTypeRegistry] = None


def get_memory_type_registry() -> MemoryTypeRegistry:
    global _registry
    if _registry is None:
        _registry = MemoryTypeRegistry()
    return _registry
