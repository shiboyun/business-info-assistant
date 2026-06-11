"""
长期记忆服务 - Long-term Memory Service

功能：
1. 对话历史压缩和总结
2. 记忆向量化存储到 Milvus
3. 记忆检索和召回
4. 用户偏好学习
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from models.chat import ChatSession, ChatMessage, LongTermMemory
from service.embedding_service import generate_embedding
from service.milvus_service import get_milvus_service, MilvusService

# 记忆触发阈值
MEMORY_TOKEN_THRESHOLD = 10000  # 超过此 token 数触发记忆压缩
MEMORY_COLLECTION_NAME = "long_term_memories"  # Milvus 集合名称


class MemoryService:
    """长期记忆服务"""

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._milvus: Optional[MilvusService] = None

    @property
    def milvus(self) -> MilvusService:
        """懒加载 Milvus 服务"""
        if self._milvus is None:
            self._milvus = get_milvus_service()
            self._ensure_memory_collection()
        return self._milvus

    def _ensure_memory_collection(self):
        """确保记忆集合存在（使用专门的 schema）"""
        from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

        collection_name = MEMORY_COLLECTION_NAME

        if utility.has_collection(collection_name):
            return

        # 定义记忆专用字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=32),  # summary/insight/preference
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),  # JSON 格式的额外信息
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]

        schema = CollectionSchema(fields=fields, description="Long-term memories")
        collection = Collection(name=collection_name, schema=schema)

        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="vector", index_params=index_params)
        collection.load()

        print(f"记忆集合 {collection_name} 创建成功")

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量（简单估算：中文约2字符/token，英文约4字符/token）"""
        # 简单估算：假设平均每3个字符为1个 token
        return len(text) // 3

    def should_compress(self, messages: List[ChatMessage]) -> bool:
        """判断是否需要压缩记忆"""
        total_tokens = sum(self.estimate_tokens(msg.content) for msg in messages)
        return total_tokens > MEMORY_TOKEN_THRESHOLD

    def summarize_conversation(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        使用 LLM 按记忆类型结构化总结对话。

        Returns:
            {
                "preferences": {...},
                "memories": [
                    {
                        "memory_type": "research_finding",
                        "action": "write",       # write / edit / delete
                        "uri": "",
                        "page_id": 1,
                        "fields": {...},
                        "links": [{"to_page_id": 2, "link_type": "derived_from", "description": "..."}]
                    }
                ],
                "topics": [...]
            }
        """
        conversation_text = "\n".join([
            f"{'用户' if msg.role == 'user' else '助手'}: {msg.content}"
            for msg in messages
        ])

        if len(conversation_text) > 30000:
            conversation_text = conversation_text[:30000] + "\n...(对话过长，已截断)"

        # 获取已注册的记忆类型说明
        from service.memory_type_registry import get_memory_type_registry
        registry = get_memory_type_registry()
        type_descriptions = []
        for schema in registry.list_all():
            fields_desc = "\n".join([
                f"      - {f.name} ({f.merge_op.value}): {f.description}"
                for f in schema.fields
            ])
            type_descriptions.append(
                f"  {schema.memory_type}: {schema.description}\n    字段:\n{fields_desc}"
            )
        types_text = "\n".join(type_descriptions) if type_descriptions else "暂无自定义类型"

        prompt = f"""请分析以下对话，提取可跨会话复用的研究成果和行业实体信息。

对话内容：
{conversation_text}

可用的记忆类型：
{types_text}

操作说明：
- action=write: 新建记忆，不需要 uri
- action=edit: 更新已有记忆，需要提供 uri
- action=delete: 删除过时或被推翻的记忆，需要提供 uri

链接说明（可选）：
- 每条记忆可包含 links 列表，引用同批次其他记忆的 page_id
- link_type 可选: derived_from（推导自）/ related_to（相关）/ contradicts（矛盾）
 
字段合并策略：
- immutable: 创建后不可修改（如 topic、entity_name），write 时提供初始值
- patch: 增量更新 - 用 SEARCH/REPLACE 格式描述变更
- replace: 全量替换
- sum: 追加到末尾

请输出以下 JSON 格式（不要包含```json标记）：
{{
    "preferences": {{
        "interests": ["用户感兴趣的领域"],
        "communication_style": "用户偏好的沟通风格",
        "focus_areas": ["用户关注的重点领域"]
    }},
    "memories": [
        {{
            "memory_type": "research_finding" 或 "industry_entity",
            "action": "write" 或 "edit" 或 "delete",
            "uri": "",
            "page_id": 1,
            "fields": {{
                "topic": "研究主题",
                "conclusion": "核心结论",
                "confidence": "高/中/低",
                "sources": "信息来源",
                "related_entities": "相关实体"
            }},
            "links": []
        }}
    ],
    "topics": ["主题标签"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的行业研究分析助手，擅长从对话中提取可复用的研究成果和行业实体信息。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )

            result_text = response.choices[0].message.content.strip()

            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result = json.loads(result_text.strip())

            # 确保兼容旧格式
            if "memories" not in result:
                result["memories"] = []
            if "preferences" not in result:
                result["preferences"] = {}
            if "topics" not in result:
                result["topics"] = []

            return result

        except Exception as e:
            print(f"总结对话失败: {e}")
            return {
                "preferences": {},
                "memories": [],
                "topics": []
            }

    def create_memory(
        self,
        db: Session,
        user_id: str,
        session_id: str,
        messages: List[ChatMessage]
    ) -> List[LongTermMemory]:
        """
        创建长期记忆 — 支持结构化输出、增量合并和链接解析。

        Returns:
            创建/更新的记忆列表
        """
        if not messages:
            return []

        # 1. 结构化总结
        summary_data = self.summarize_conversation(messages)
        memories_data = summary_data.get("memories", [])
        preferences_data = summary_data.get("preferences", {})
        topics_data = summary_data.get("topics", [])

        total_tokens = sum(self.estimate_tokens(msg.content) for msg in messages)
        created_memories: List[LongTermMemory] = []

        # 2. page_id → 真实 URI 映射表（用于链接解析）
        page_id_to_uri: dict[int, str] = {}

        # 3. 处理每条结构化记忆
        for mem_data in memories_data:
            memory_type = mem_data.get("memory_type", "")
            action = mem_data.get("action", "write")
            fields = mem_data.get("fields", {})
            page_id = mem_data.get("page_id", 0)
            uri = mem_data.get("uri", "")
            links = mem_data.get("links", [])

            from service.memory_type_registry import get_memory_type_registry
            registry = get_memory_type_registry()
            schema = registry.get(memory_type)

            if schema is None:
                print(f"未知的记忆类型: {memory_type}，跳过")
                continue

            if action == "delete" and uri:
                self._delete_by_uri(db, user_id, uri)
                continue

            if action == "edit" and uri:
                existing = self._find_by_uri(db, user_id, uri)
            else:
                existing = self._find_existing(db, user_id, memory_type, fields, schema)

            if existing:
                existing.fields = self._merge_fields(
                    existing.fields or {}, fields, schema
                )
                existing.summary = fields.get("conclusion") or fields.get("key_facts") or existing.summary
                existing.abstract = self._generate_abstract(existing.summary)
                existing.overview = self._generate_overview(memory_type, existing.fields)
                existing.token_count = max(existing.token_count or 0, total_tokens)
                existing.key_insights = {** (existing.key_insights or {}), **preferences_data}
                existing.links = self._merge_links(existing.links or [], links, page_id_to_uri)
                db.commit()
                db.refresh(existing)
                created_memories.append(existing)
                if page_id:
                    page_id_to_uri[page_id] = str(existing.id)
            else:
                abstract = self._generate_abstract(
                    fields.get("conclusion") or fields.get("key_facts") or ""
                )
                overview = self._generate_overview(memory_type, fields)

                memory = LongTermMemory(
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=memory_type,
                    summary=fields.get("conclusion") or fields.get("key_facts") or "",
                    abstract=abstract,
                    overview=overview,
                    key_insights=preferences_data,
                    fields=fields,
                    links=[],
                    backlinks=[],
                    token_count=total_tokens,
                )
                db.add(memory)
                db.commit()
                db.refresh(memory)
                created_memories.append(memory)
                if page_id:
                    page_id_to_uri[page_id] = str(memory.id)

        # 4. 解析链接
        for memory in created_memories:
            if memory.links:
                resolved_links = []
                for link in memory.links:
                    to_page_id = link.get("to_page_id")
                    if to_page_id and to_page_id in page_id_to_uri:
                        link["to_uri"] = page_id_to_uri[to_page_id]
                        link.pop("to_page_id", None)
                        resolved_links.append(link)
                memory.links = resolved_links
                db.commit()
                self._update_backlinks(db, str(memory.id), memory.links)

        # 5. 处理 preferences（仅当无结构化记忆时单独存一条）
        if not created_memories and preferences_data:
            memory = LongTermMemory(
                user_id=user_id,
                session_id=session_id,
                memory_type="preference",
                summary=json.dumps(preferences_data, ensure_ascii=False),
                abstract=f"用户偏好: {', '.join(preferences_data.get('interests', []))}",
                overview=f"关注领域: {', '.join(preferences_data.get('focus_areas', []))}",
                key_insights=preferences_data,
                fields=preferences_data,
                links=[],
                backlinks=[],
                token_count=total_tokens,
            )
            db.add(memory)
            db.commit()
            db.refresh(memory)
            created_memories.append(memory)

        # 6. 向量化
        for memory in created_memories:
            milvus_ids = self._store_memory_vectors(
                memory_id=str(memory.id),
                user_id=user_id,
                session_id=session_id,
                memory_type=memory.memory_type,
                summary_data={
                    "abstract": memory.abstract or "",
                    "overview": memory.overview or "",
                    "summary": memory.summary or "",
                    "topics": topics_data,
                    "fields": memory.fields or {},
                }
            )
            memory.milvus_ids = milvus_ids
            db.commit()

        print(f"创建/更新了 {len(created_memories)} 条长期记忆")
        return created_memories

    def _generate_abstract(self, text: str) -> str:
        """从长文本生成 L0 摘要（取前 2-3 句，约 50 字）。"""
        if not text:
            return ""
        truncated = text[:100]
        last_period = max(truncated.rfind("。"), truncated.rfind("."), truncated.rfind("；"))
        if last_period > 20:
            return truncated[:last_period + 1]
        return truncated

    def _generate_overview(self, memory_type: str, fields: dict) -> str:
        """从 fields 生成 L1 概览（约 150 字）。"""
        parts = [f"[{memory_type}]"]
        for key, value in fields.items():
            if key in ("conclusion", "key_facts"):
                parts.append(str(value)[:120])
            elif key in ("entity_name", "topic"):
                parts.append(f"{key}: {value}")
        return " | ".join(parts)

    def _merge_fields(
        self, existing_fields: dict, new_fields: dict, schema
    ) -> dict:
        """按 schema 中定义的 merge_op 合并字段。"""
        from service.memory_merge import apply_merge

        merged = dict(existing_fields)
        for field_def in schema.fields:
            field_name = field_def.name
            if field_name in new_fields:
                current = merged.get(field_name)
                merged[field_name] = apply_merge(current, new_fields[field_name], field_def.merge_op)
        for key, value in new_fields.items():
            if key not in merged:
                merged[key] = value
        return merged

    def _find_existing(
        self, db: Session, user_id: str, memory_type: str, fields: dict, schema
    ) -> Optional[LongTermMemory]:
        """根据 immutable 字段查找已存在的记忆。"""
        immutable_keys = [
            f.name for f in schema.fields
            if f.merge_op.value == "immutable" and f.name in fields
        ]
        for key in immutable_keys:
            existing = db.query(LongTermMemory).filter(
                LongTermMemory.user_id == user_id,
                LongTermMemory.memory_type == memory_type,
                LongTermMemory.fields[key].astext == fields[key]
            ).first()
            if existing:
                return existing
        return None

    def _find_by_uri(self, db: Session, user_id: str, uri: str) -> Optional[LongTermMemory]:
        """通过 URI（即 memory id）查找。"""
        try:
            from uuid import UUID
            mem_uuid = UUID(uri)
            return db.query(LongTermMemory).filter(
                LongTermMemory.id == mem_uuid,
                LongTermMemory.user_id == user_id,
            ).first()
        except (ValueError, AttributeError):
            return None

    def _delete_by_uri(self, db: Session, user_id: str, uri: str) -> None:
        """删除指定记忆。"""
        memory = self._find_by_uri(db, user_id, uri)
        if memory:
            self.delete_memory(db, str(memory.id), user_id)

    def _merge_links(
        self, existing_links: list, new_links: list, page_id_to_uri: dict
    ) -> list:
        """合并链接列表，按 to_page_id 去重。"""
        import json as _json
        merged = list(existing_links)
        seen = {_json.dumps(l, sort_keys=True, ensure_ascii=False) for l in merged}
        for link in new_links:
            link_copy = dict(link)
            serialized = _json.dumps(link_copy, sort_keys=True, ensure_ascii=False)
            if serialized not in seen:
                merged.append(link_copy)
                seen.add(serialized)
        return merged

    def _update_backlinks(self, db: Session, from_id: str, links: list) -> None:
        """更新目标记忆的 backlinks 字段。"""
        for link in links:
            to_uri = link.get("to_uri")
            if not to_uri:
                continue
            try:
                from uuid import UUID
                target = db.query(LongTermMemory).filter(
                    LongTermMemory.id == UUID(to_uri)
                ).first()
                if target:
                    backlinks = list(target.backlinks or [])
                    new_backlink = {
                        "from_uri": from_id,
                        "link_type": link.get("link_type", "related_to"),
                    }
                    if not any(
                        b.get("from_uri") == from_id
                        for b in backlinks
                    ):
                        backlinks.append(new_backlink)
                        target.backlinks = backlinks
                        db.commit()
            except (ValueError, AttributeError):
                pass

    def _store_memory_vectors(
        self,
        memory_id: str,
        user_id: str,
        session_id: str,
        memory_type: str = "preference",
        summary_data: Dict[str, Any] = None,
    ) -> List[str]:
        """将记忆内容向量化并存储到 Milvus。"""
        from pymilvus import Collection

        if summary_data is None:
            summary_data = {}

        milvus_ids = []
        documents_to_insert = []

        from service.memory_type_registry import get_memory_type_registry
        registry = get_memory_type_registry()
        schema = registry.get(memory_type)

        # 1. L0 abstract 向量
        abstract = summary_data.get("abstract", "")
        if abstract:
            abstract_vector = generate_embedding(abstract)
            if abstract_vector:
                doc_id = f"{memory_id}_abstract"
                documents_to_insert.append({
                    "id": doc_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "content": abstract[:65535],
                    "metadata": json.dumps({"memory_id": memory_id, "level": "abstract"}),
                    "vector": abstract_vector
                })
                milvus_ids.append(doc_id)

        # 2. L2 summary 向量（主检索向量）
        summary = summary_data.get("summary", "")
        if summary:
            if schema and schema.embedding_template:
                embedding_text = schema.embedding_template
                for key in ("topic", "conclusion", "entity_name", "industry", "key_facts"):
                    embedding_text = embedding_text.replace(
                        f"{{{key}}}", summary_data.get("fields", {}).get(key, "")
                    )
            else:
                embedding_text = summary

            summary_vector = generate_embedding(embedding_text)
            if summary_vector:
                doc_id = f"{memory_id}_summary"
                documents_to_insert.append({
                    "id": doc_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "content": summary[:65535],
                    "metadata": json.dumps({"memory_id": memory_id, "level": "detail"}),
                    "vector": summary_vector
                })
                milvus_ids.append(doc_id)

        # 3. Topics 向量
        topics = summary_data.get("topics", [])
        if topics:
            topics_text = "研究主题: " + ", ".join(topics)
            topics_vector = generate_embedding(topics_text)
            if topics_vector:
                doc_id = f"{memory_id}_topics"
                documents_to_insert.append({
                    "id": doc_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "content": topics_text[:65535],
                    "metadata": json.dumps({"memory_id": memory_id, "topics": topics}),
                    "vector": topics_vector
                })
                milvus_ids.append(doc_id)

        # 批量插入 Milvus
        if documents_to_insert:
            try:
                collection = Collection(MEMORY_COLLECTION_NAME)
                collection.load()
                ids = [doc["id"] for doc in documents_to_insert]
                user_ids = [doc["user_id"] for doc in documents_to_insert]
                session_ids = [doc["session_id"] for doc in documents_to_insert]
                memory_types_list = [doc["memory_type"] for doc in documents_to_insert]
                contents = [doc["content"][:65535] for doc in documents_to_insert]
                metadatas = [doc["metadata"][:8192] for doc in documents_to_insert]
                vectors = [doc["vector"] for doc in documents_to_insert]

                data = [ids, user_ids, session_ids, memory_types_list, contents, metadatas, vectors]
                collection.insert(data)
                collection.flush()
                print(f"成功存储 {len(documents_to_insert)} 条记忆向量")
            except Exception as e:
                print(f"存储记忆向量失败: {e}")

        return milvus_ids

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        检索与查询相关的记忆

        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关记忆列表
        """
        from pymilvus import Collection, utility

        if not utility.has_collection(MEMORY_COLLECTION_NAME):
            return []

        # 生成查询向量
        query_vector = generate_embedding(query)
        if not query_vector:
            return []

        try:
            collection = Collection(MEMORY_COLLECTION_NAME)
            collection.load()

            # 按用户过滤
            expr = f'user_id == "{user_id}"'

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "session_id", "memory_type", "content", "metadata"],
            )

            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("id"),
                        "session_id": hit.entity.get("session_id"),
                        "memory_type": hit.entity.get("memory_type"),
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score,
                    })

            return formatted_results

        except Exception as e:
            print(f"检索记忆失败: {e}")
            return []

    def get_user_memories(
        self,
        db: Session,
        user_id: str,
        limit: int = 10
    ) -> List[LongTermMemory]:
        """获取用户的所有长期记忆"""
        return db.query(LongTermMemory).filter(
            LongTermMemory.user_id == user_id
        ).order_by(LongTermMemory.created_at.desc()).limit(limit).all()

    def delete_memory(
        self,
        db: Session,
        memory_id: str,
        user_id: str
    ) -> bool:
        """删除指定的长期记忆"""
        from pymilvus import Collection, utility

        memory = db.query(LongTermMemory).filter(
            LongTermMemory.id == memory_id,
            LongTermMemory.user_id == user_id
        ).first()

        if not memory:
            return False

        # 删除 Milvus 中的向量
        if memory.milvus_ids and utility.has_collection(MEMORY_COLLECTION_NAME):
            try:
                collection = Collection(MEMORY_COLLECTION_NAME)
                for milvus_id in memory.milvus_ids:
                    expr = f'id == "{milvus_id}"'
                    collection.delete(expr)
            except Exception as e:
                print(f"删除 Milvus 记忆失败: {e}")

        # 删除数据库记录
        db.delete(memory)
        db.commit()

        return True

    def build_memory_context(
        self,
        user_id: str,
        current_query: str,
        max_memories: int = 5,
        max_tokens: int = 800,
    ) -> dict:
        """
        构建记忆上下文，用于增强当前对话。三层分级加载。

        Args:
            user_id: 用户ID
            current_query: 当前查询
            max_memories: 最大记忆数量（L1 overview 级别）
            max_tokens: context_text 最大 token 数

        Returns:
            {"context_text": "...", "memory_ids": ["id1", ...]}
        """
        # Step 1: 粗筛 — Milvus 搜索 top-15
        all_results = self.retrieve_memories(user_id, current_query, top_k=15)

        if not all_results:
            return {"context_text": "", "memory_ids": []}

        # Step 2: 去重 + 按 memory_type 分组（每种最多 3 条）
        from collections import defaultdict
        by_type: dict[str, list] = defaultdict(list)
        seen_ids = set()
        for r in all_results:
            mem_id = r.get("metadata", "{}")
            try:
                meta = json.loads(mem_id) if isinstance(mem_id, str) else mem_id
                real_id = meta.get("memory_id") or r.get("id", "")
            except (json.JSONDecodeError, TypeError):
                real_id = r.get("id", "").rsplit("_", 1)[0]

            if real_id in seen_ids:
                continue
            seen_ids.add(real_id)

            mem_type = r.get("memory_type", "preference")
            if len(by_type[mem_type]) < 3:
                by_type[mem_type].append({
                    "id": real_id,
                    "content": r.get("content", ""),
                    "memory_type": mem_type,
                    "score": r.get("score", 0),
                })

        # Step 3: 从 DB 加载 L1 overview，构建 context_text
        candidates = []
        for items in by_type.values():
            candidates.extend(items)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:max_memories]

        if not candidates:
            return {"context_text": "", "memory_ids": []}

        from models.chat import LongTermMemory
        from uuid import UUID

        context_parts = ["[相关历史研究记忆]"]
        memory_ids = []
        total_chars = 0

        for c in candidates:
            try:
                mem_uuid = UUID(c["id"])
                mem = self._db_query_by_id(mem_uuid)
            except (ValueError, AttributeError):
                mem = None

            if mem and mem.overview:
                overview = f"- [{mem.memory_type}] {mem.overview}"
            else:
                overview = f"- [{c['memory_type']}] {c['content'][:120]}"

            if total_chars + len(overview) > max_tokens * 3:
                break

            context_parts.append(overview)
            memory_ids.append(c["id"])
            total_chars += len(overview)

        context_parts.append("")
        context_text = "\n".join(context_parts)

        return {
            "context_text": context_text,
            "memory_ids": memory_ids,
        }

    def _db_query_by_id(self, mem_uuid) -> Optional[Any]:
        """从数据库查询单条记忆（内部方法）。"""
        from core.database import SessionLocal
        from models.chat import LongTermMemory
        db = SessionLocal()
        try:
            return db.query(LongTermMemory).filter(
                LongTermMemory.id == mem_uuid
            ).first()
        finally:
            db.close()


# 单例实例
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """获取记忆服务单例"""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
