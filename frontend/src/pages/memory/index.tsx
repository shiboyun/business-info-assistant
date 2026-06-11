import { useEffect, useState, useCallback } from 'react'
import {
  Card,
  Button,
  List,
  Input,
  message,
  Popconfirm,
  Tag,
  Empty,
  Spin,
  Typography,
  Collapse,
  Modal,
  Select,
} from 'antd'
import {
  DeleteOutlined,
  SearchOutlined,
  BulbOutlined,
  ClockCircleOutlined,
  FileTextOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { useSnapshot } from 'valtio'
import { authState } from '@/store/auth'
import { useNavigate } from 'react-router-dom'
import * as api from '@/api'
import type { Memory, MemorySearchResult } from '@/api/memory'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import 'dayjs/locale/zh-cn'
import styles from './index.module.scss'

dayjs.extend(relativeTime)
dayjs.locale('zh-cn')

const { Text, Paragraph } = Typography
const { Search } = Input

const MEMORY_TYPE_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  research_finding: { label: '研究发现', color: '#1677ff', icon: <BulbOutlined /> },
  industry_entity: { label: '行业实体', color: '#52c41a', icon: <FileTextOutlined /> },
  preference: { label: '用户偏好', color: '#fa8c16', icon: <ClockCircleOutlined /> },
  general: { label: '通用', color: '#8c8c8c', icon: <BulbOutlined /> },
}

const LINK_TYPE_LABELS: Record<string, string> = {
  derived_from: '来源于',
  related_to: '关联到',
  contradicts: '矛盾于',
}

function getMemoryTypeConfig(type: string) {
  return MEMORY_TYPE_CONFIG[type] || MEMORY_TYPE_CONFIG.general
}

function formatFieldLabel(key: string): string {
  const labels: Record<string, string> = {
    topic: '研究主题',
    conclusion: '核心结论',
    confidence: '置信度',
    sources: '信息来源',
    related_entities: '相关实体',
    entity_name: '实体名称',
    entity_type: '实体类型',
    industry: '所属行业',
    key_facts: '关键事实',
    last_researched: '最近研究时间',
    preference_type: '偏好类型',
    value: '偏好值',
  }
  return labels[key] || key.replace(/_/g, ' ')
}

export default function MemoryPage() {
  const navigate = useNavigate()
  const { isLoggedIn } = useSnapshot(authState)
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(false)
  const [searchResults, setSearchResults] = useState<MemorySearchResult[] | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [total, setTotal] = useState(0)

  // 生成记忆弹窗
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [sessions, setSessions] = useState<api.session.Session[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [createLoading, setCreateLoading] = useState(false)

  const fetchMemories = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.memory.getMemories({ limit: 50 })
      if (res.data) {
        setMemories(res.data.memories)
        setTotal(res.data.total)
      }
    } catch (error: any) {
      message.error(error?.response?.data?.detail || '获取记忆列表失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isLoggedIn) {
      fetchMemories()
    }
  }, [isLoggedIn, fetchMemories])

  if (!isLoggedIn) {
    return (
      <div className={styles['memory-page']}>
        <div className={styles['empty-state']}>
          <Empty description="请先登录" />
          <Button type="primary" onClick={() => navigate('/login')}>
            去登录
          </Button>
        </div>
      </div>
    )
  }

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults(null)
      return
    }
    setSearchLoading(true)
    try {
      const res = await api.memory.searchMemories({ query, top_k: 10 })
      if (res.data) {
        setSearchResults(res.data)
      }
    } catch (error: any) {
      message.error(error?.response?.data?.detail || '搜索失败')
    } finally {
      setSearchLoading(false)
    }
  }

  const handleDelete = async (memoryId: string) => {
    try {
      await api.memory.deleteMemory(memoryId)
      message.success('记忆已删除')
      fetchMemories()
      // 如果正在显示搜索结果，也需要更新
      if (searchResults) {
        setSearchResults(searchResults.filter(r => r.id !== memoryId))
      }
    } catch (error: any) {
      message.error(error?.response?.data?.detail || '删除失败')
    }
  }

  const handleClearSearch = () => {
    setSearchResults(null)
  }

  const handleOpenCreateModal = async () => {
    setCreateModalOpen(true)
    setSessionsLoading(true)
    setSelectedSessionId(null)
    try {
      const res = await api.session.getSessions({ limit: 20 })
      if (res.data) {
        setSessions(res.data)
      }
    } catch {
      message.error('获取会话列表失败')
    } finally {
      setSessionsLoading(false)
    }
  }

  const handleCreateMemory = async () => {
    if (!selectedSessionId) {
      message.warning('请选择要生成记忆的会话')
      return
    }
    setCreateLoading(true)
    try {
      const res = await api.memory.createMemory(selectedSessionId)
      if (res.data) {
        message.success(`成功生成记忆: ${(res.data as any).memory_type || ''}`)
        setCreateModalOpen(false)
        fetchMemories()
      }
    } catch (error: any) {
      message.error(error?.response?.data?.detail || '生成记忆失败')
    } finally {
      setCreateLoading(false)
    }
  }

  const renderMemoryList = () => (
    <List
      className={styles['memory-list']}
      loading={loading}
      dataSource={memories}
      locale={{ emptyText: <Empty description="暂无记忆" /> }}
      renderItem={(memory) => {
        const typeConfig = getMemoryTypeConfig(memory.memory_type)
        return (
          <List.Item
            className={styles['memory-item']}
            actions={[
              <Popconfirm
                key="delete"
                title="确定删除此记忆？"
                description="删除后将无法恢复"
                onConfirm={() => handleDelete(memory.id)}
              >
                <Button type="text" danger size="small" icon={<DeleteOutlined />} />
              </Popconfirm>,
            ]}
          >
            <List.Item.Meta
              avatar={
                <div className={styles['memory-icon']} style={{ background: `linear-gradient(135deg, ${typeConfig.color}20, ${typeConfig.color}40)` }}>
                  <span style={{ color: typeConfig.color }}>{typeConfig.icon}</span>
                </div>
              }
              title={
                <div className={styles['memory-title']}>
                  <Tag color={typeConfig.color}>{typeConfig.label}</Tag>
                  <Text ellipsis={{ tooltip: true }} style={{ flex: 1 }}>
                    {memory.abstract || memory.summary}
                  </Text>
                  {memory.token_count && (
                    <Tag color="blue">{memory.token_count} tokens</Tag>
                  )}
                </div>
              }
              description={
                <div className={styles['memory-meta']}>
                  {memory.abstract && memory.summary && (
                    <Paragraph
                      ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
                      className={styles['memory-summary']}
                      type="secondary"
                    >
                      {memory.summary}
                    </Paragraph>
                  )}
                  {memory.overview && (
                    <div className={styles['memory-overview']}>
                      <Text type="secondary" italic style={{ fontSize: 12 }}>
                        {memory.overview}
                      </Text>
                    </div>
                  )}
                  {memory.fields && Object.keys(memory.fields).length > 0 && (
                    <Collapse
                      ghost
                      size="small"
                      className={styles['fields-collapse']}
                      items={[{
                        key: 'fields',
                        label: <Text type="secondary" style={{ fontSize: 12 }}>字段详情</Text>,
                        children: (
                          <div className={styles['fields-grid']}>
                            {Object.entries(memory.fields).map(([key, value]) => (
                              <div key={key} className={styles['field-item']}>
                                <Text type="secondary" style={{ fontSize: 11 }}>{formatFieldLabel(key)}</Text>
                                <Text style={{ fontSize: 13 }}>{String(value)}</Text>
                              </div>
                            ))}
                          </div>
                        ),
                      }]}
                    />
                  )}
                  {memory.links && memory.links.length > 0 && (
                    <div className={styles['memory-links']}>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        关联: {memory.links.map(l => LINK_TYPE_LABELS[l.link_type] || l.link_type).join(' · ')}
                      </Text>
                    </div>
                  )}
                  <div className={styles['memory-footer']}>
                    <span>
                      <ClockCircleOutlined style={{ marginRight: 4 }} />
                      {dayjs(memory.created_at).format('YYYY-MM-DD HH:mm')}
                    </span>
                    <span>{dayjs(memory.created_at).fromNow()}</span>
                  </div>
                </div>
              }
            />
          </List.Item>
        )
      }}
    />
  )

  const renderSearchResults = () => (
    <div className={styles['search-results']}>
      <div className={styles['search-header']}>
        <Text type="secondary">找到 {searchResults?.length || 0} 条相关记忆</Text>
        <Button type="link" onClick={handleClearSearch}>
          清除搜索
        </Button>
      </div>
      <List
        className={styles['memory-list']}
        loading={searchLoading}
        dataSource={searchResults || []}
        locale={{ emptyText: <Empty description="未找到相关记忆" /> }}
        renderItem={(result) => {
          const typeConfig = getMemoryTypeConfig(result.memory_type)
          return (
            <List.Item
              className={styles['memory-item']}
              actions={[
                <Popconfirm
                  key="delete"
                  title="确定删除此记忆？"
                  onConfirm={() => handleDelete(result.id)}
                >
                  <Button type="text" danger size="small" icon={<DeleteOutlined />} />
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                avatar={
                  <div className={styles['memory-icon']} style={{ background: `linear-gradient(135deg, ${typeConfig.color}20, ${typeConfig.color}40)` }}>
                    <span style={{ color: typeConfig.color }}>{typeConfig.icon}</span>
                  </div>
                }
                title={
                  <div className={styles['memory-title']}>
                    <Tag color={typeConfig.color}>{typeConfig.label}</Tag>
                    <Tag color="purple">相关度 {(result.score * 100).toFixed(0)}%</Tag>
                    <Text ellipsis={{ tooltip: true }} style={{ flex: 1 }}>
                      {result.content}
                    </Text>
                  </div>
                }
              />
            </List.Item>
          )
        }}
      />
    </div>
  )

  return (
    <div className={styles['memory-page']}>
      <div className={styles['header']}>
        <div className={styles['header-left']}>
          <h2>记忆库</h2>
          <Text type="secondary">共 {total} 条记忆</Text>
        </div>
        <div className={styles['header-right']}>
          <Button icon={<ThunderboltOutlined />} onClick={handleOpenCreateModal}>
            生成记忆
          </Button>
          <Button icon={<ReloadOutlined />} onClick={fetchMemories} loading={loading}>
            刷新
          </Button>
        </div>
      </div>

      <div className={styles['search-bar']}>
        <Search
          placeholder="搜索相关记忆..."
          allowClear
          enterButton={<SearchOutlined />}
          size="large"
          onSearch={handleSearch}
          loading={searchLoading}
          style={{ maxWidth: 600 }}
        />
      </div>

      <Card className={styles['content-card']}>
        {searchResults ? renderSearchResults() : renderMemoryList()}
      </Card>

      <Modal
        title="从历史会话生成记忆"
        open={createModalOpen}
        onOk={handleCreateMemory}
        onCancel={() => setCreateModalOpen(false)}
        confirmLoading={createLoading}
        okText="生成"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <Text type="secondary">
            选择一条历史会话，AI 将自动提取其中的研究成果和行业实体，生成结构化长期记忆。
          </Text>
        </div>
        <Select
          showSearch
          style={{ width: '100%' }}
          placeholder="选择会话..."
          loading={sessionsLoading}
          value={selectedSessionId}
          onChange={(val) => setSelectedSessionId(val)}
          filterOption={(input, option) =>
            (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
          }
          options={sessions.map((s) => ({
            label: `${s.title || s.id} (${s.message_count} 条消息)`,
            value: s.id,
          }))}
        />
      </Modal>
    </div>
  )
}
