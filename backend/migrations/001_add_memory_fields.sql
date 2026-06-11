-- 记忆层升级 V2: 新增记忆类型、三层摘要、动态字段、链接关系
ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS memory_type VARCHAR(64) DEFAULT 'general';
ALTER TABLE long_term_memories ALTER COLUMN memory_type SET DEFAULT 'general';
ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS abstract TEXT;
ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS overview TEXT;
ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS fields JSONB DEFAULT '{}'::jsonb;
ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS links JSONB DEFAULT '[]'::jsonb;
ALTER TABLE long_term_memories ADD COLUMN IF NOT EXISTS backlinks JSONB DEFAULT '[]'::jsonb;
CREATE INDEX IF NOT EXISTS idx_long_term_memories_memory_type ON long_term_memories(memory_type);
