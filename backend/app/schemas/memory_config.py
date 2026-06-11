"""记忆类型配置 Pydantic Schema"""
from enum import Enum

from pydantic import BaseModel, Field


class MergeOp(str, Enum):
    PATCH = "patch"
    REPLACE = "replace"
    SUM = "sum"
    IMMUTABLE = "immutable"


class FieldType(str, Enum):
    STRING = "string"


class MemoryField(BaseModel):
    name: str
    type: FieldType = FieldType.STRING
    description: str = ""
    merge_op: MergeOp = MergeOp.PATCH


class MemoryTypeSchema(BaseModel):
    memory_type: str
    description: str = ""
    directory: str = ""
    filename_template: str = ""
    embedding_template: str = ""
    fields: list[MemoryField] = Field(default_factory=list)
