import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import {
  ApiError,
  createKnowledgeBase,
  deleteChatSession,
  deleteDocument,
  fetchDocumentFile,
  getChatSession,
  getHealth,
  listChatSessions,
  listDocuments,
  listKnowledgeBases,
  sendChatMessage,
  uploadDocuments,
  type ServiceHealth,
} from './api'
import { Icon } from './Icon'
import type {
  AppView,
  ChatMessage,
  ChatSession,
  Citation,
  DocumentStatus,
  KnowledgeBase,
  KnowledgeDocument,
  UploadItem,
} from './types'

const STATUS_LABELS: Record<DocumentStatus, string> = {
  queued: '待上传',
  uploading: '上传中',
  processing: '处理中',
  parsing: '解析中',
  chunking: '切分中',
  embedding: '向量化',
  indexing: '写入索引',
  ready: '可用',
  failed: '失败',
}

const SERVICE_HEALTH_COPY: Record<ServiceHealth | 'checking', { label: string; title: string }> = {
  checking: { label: '正在连接', title: '正在检查 API 服务' },
  online: { label: '服务正常', title: 'API 与向量服务正常' },
  degraded: { label: '服务受限', title: 'API 可访问，但部分依赖服务不可用' },
  offline: { label: '服务未连接', title: 'API 服务不可用' },
}

function formatBytes(bytes?: number) {
  if (bytes === undefined || bytes === null) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatDate(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function relevancePercent(score: number) {
  if (!Number.isFinite(score)) return 0
  return Math.round(Math.min(1, Math.max(0, score)) * 100)
}

function idString(value: unknown) {
  return String(value ?? '')
}

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return
    } catch {
      // Fall back for non-secure origins and restrictive browser policies.
    }
  }

  const field = document.createElement('textarea')
  field.value = text
  field.setAttribute('readonly', '')
  field.style.position = 'fixed'
  field.style.opacity = '0'
  document.body.appendChild(field)
  field.select()
  const copied = document.execCommand('copy')
  field.remove()
  if (!copied) throw new Error('浏览器未授予剪贴板权限')
}

function statusIcon(status: DocumentStatus) {
  if (status === 'ready') return 'check' as const
  if (status === 'failed') return 'alert' as const
  if (status === 'queued') return 'clock' as const
  return 'loader' as const
}

function StatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span className={`status-badge status-${status}`}>
      <Icon name={statusIcon(status)} size={13} className={status !== 'ready' && status !== 'failed' ? 'spin' : ''} />
      {STATUS_LABELS[status]}
    </span>
  )
}

function EmptyBlock({
  icon,
  title,
  body,
  action,
}: {
  icon: 'chat' | 'file' | 'library' | 'quote' | 'route'
  title: string
  body: string
  action?: React.ReactNode
}) {
  return (
    <div className="empty-block">
      <span className="empty-icon"><Icon name={icon} size={24} /></span>
      <strong>{title}</strong>
      <p>{body}</p>
      {action}
    </div>
  )
}

function SessionList({
  sessions,
  activeSessionId,
  deletingSessionId,
  sending,
  onOpenSession,
  onRequestDelete,
}: {
  sessions: ChatSession[]
  activeSessionId?: string
  deletingSessionId?: string
  sending: boolean
  onOpenSession: (session: ChatSession) => void
  onRequestDelete: (session: ChatSession) => void
}) {
  return (
    <div className="session-list" aria-busy={Boolean(deletingSessionId)}>
      {sessions.length === 0 && <p className="sidebar-empty">暂无历史对话</p>}
      {sessions.slice(0, 12).map((session) => {
        const sessionId = idString(session.id)
        const active = idString(activeSessionId) === sessionId
        const deleting = idString(deletingSessionId) === sessionId
        const title = session.title || '未命名对话'
        return (
          <div className={`session-row ${active ? 'active' : ''}`} key={sessionId}>
            <button
              type="button"
              className="session-open"
              onClick={() => onOpenSession(session)}
              title={title}
              disabled={deleting}
              aria-current={active ? 'page' : undefined}
            >
              <Icon name="chat" size={15} />
              <span className="session-copy">
                <strong>{title}</strong>
                <small>{formatDate(session.updated_at || session.created_at)}</small>
              </span>
            </button>
            <button
              type="button"
              className="session-delete"
              onClick={() => onRequestDelete(session)}
              disabled={Boolean(deletingSessionId)}
              aria-label={`删除对话：${title}`}
              title={active && sending ? '请先停止当前回答' : '删除对话'}
            >
              <Icon name={deleting ? 'loader' : 'trash'} size={14} className={deleting ? 'spin' : ''} />
            </button>
          </div>
        )
      })}
    </div>
  )
}

function AppSidebar({
  view,
  setView,
  sessions,
  activeSessionId,
  deletingSessionId,
  sending,
  onOpenSession,
  onRequestDelete,
  onShowHistory,
  historyOpen,
  onNewChat,
}: {
  view: AppView
  setView: (view: AppView) => void
  sessions: ChatSession[]
  activeSessionId?: string
  deletingSessionId?: string
  sending: boolean
  onOpenSession: (session: ChatSession) => void
  onRequestDelete: (session: ChatSession) => void
  onShowHistory: () => void
  historyOpen: boolean
  onNewChat: () => void
}) {
  return (
    <aside className="app-sidebar">
      <div className="brand">
        <span className="brand-mark"><Icon name="spark" size={21} /></span>
        <div><strong>知屿</strong><span>企业知识工作台</span></div>
      </div>

      <button className="new-chat-button" type="button" onClick={onNewChat}>
        <Icon name="plus" size={17} /> 新建对话
      </button>

      <nav className="primary-nav" aria-label="主导航">
        <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')} aria-current={view === 'chat' ? 'page' : undefined}>
          <Icon name="chat" /> 智能问答
        </button>
        <button className={view === 'knowledge' ? 'active' : ''} onClick={() => setView('knowledge')} aria-current={view === 'knowledge' ? 'page' : undefined}>
          <Icon name="library" /> 知识库
        </button>
      </nav>

      <div className="sidebar-section-title">
        <span>最近对话</span>
        {sessions.length > 0 && <span className="session-count">{Math.min(sessions.length, 99)}</span>}
      </div>
      <button className="compact-history-button" type="button" onClick={onShowHistory} aria-label="打开最近对话" aria-haspopup="dialog" aria-expanded={historyOpen} aria-controls="history-panel">
        <Icon name="clock" size={18} />
      </button>
      <SessionList
        sessions={sessions}
        activeSessionId={activeSessionId}
        deletingSessionId={deletingSessionId}
        sending={sending}
        onOpenSession={onOpenSession}
        onRequestDelete={onRequestDelete}
      />

      <div className="sidebar-footer">
        <div className="avatar">管</div>
        <div><strong>管理员</strong><span>企业工作区</span></div>
        <Icon name="chevron" size={15} />
      </div>
    </aside>
  )
}

function MobileNav({
  view,
  setView,
  onShowHistory,
  historyOpen,
  onNewChat,
}: {
  view: AppView
  setView: (view: AppView) => void
  onShowHistory: () => void
  historyOpen: boolean
  onNewChat: () => void
}) {
  return (
    <nav className="mobile-nav" aria-label="移动端主导航">
      <button type="button" className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')} aria-current={view === 'chat' ? 'page' : undefined}>
        <Icon name="chat" size={18} /><span>问答</span>
      </button>
      <button type="button" onClick={onShowHistory} aria-haspopup="dialog" aria-expanded={historyOpen} aria-controls="history-panel">
        <Icon name="clock" size={18} /><span>最近</span>
      </button>
      <button type="button" className="mobile-new-chat" onClick={onNewChat}>
        <Icon name="plus" size={19} /><span>新对话</span>
      </button>
      <button type="button" className={view === 'knowledge' ? 'active' : ''} onClick={() => setView('knowledge')} aria-current={view === 'knowledge' ? 'page' : undefined}>
        <Icon name="library" size={18} /><span>知识库</span>
      </button>
    </nav>
  )
}

function KnowledgeSelector({
  knowledgeBases,
  selected,
  onChange,
}: {
  knowledgeBases: KnowledgeBase[]
  selected: string[]
  onChange: (ids: string[]) => void
}) {
  const label = useMemo(() => {
    if (!selected.length) return '选择知识库'
    if (selected.length === 1) {
      return knowledgeBases.find((item) => idString(item.id) === selected[0])?.name || '1 个知识库'
    }
    return `${selected.length} 个知识库`
  }, [knowledgeBases, selected])

  function toggle(id: string) {
    onChange(selected.includes(id) ? selected.filter((value) => value !== id) : [...selected, id])
  }

  return (
    <details className="knowledge-selector">
      <summary><Icon name="database" size={15} /><span>{label}</span><Icon name="chevron" size={14} /></summary>
      <div className="selector-menu">
        <div className="selector-heading"><strong>回答范围</strong><span>可多选</span></div>
        {knowledgeBases.length === 0 ? (
          <p>暂无可用知识库</p>
        ) : knowledgeBases.map((item) => {
          const id = idString(item.id)
          return (
            <label key={id}>
              <input type="checkbox" checked={selected.includes(id)} onChange={() => toggle(id)} />
              <span className="checkbox-mark"><Icon name="check" size={12} /></span>
              <span><strong>{item.name}</strong><small>{item.document_count ?? 0} 份文档</small></span>
            </label>
          )
        })}
      </div>
    </details>
  )
}

function MessageBubble({
  message,
  onCitation,
}: {
  message: ChatMessage
  onCitation: (citation: Citation) => void
}) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')

  async function copyAnswer() {
    try {
      await copyText(message.content)
      setCopyState('copied')
    } catch {
      setCopyState('failed')
    }
    window.setTimeout(() => setCopyState('idle'), 1800)
  }

  if (message.role === 'user') {
    return (
      <article className="message-row user-message">
        <div className="message-avatar user"><Icon name="user" size={17} /></div>
        <div className="message-body"><div className="user-bubble">{message.content}</div></div>
      </article>
    )
  }

  return (
    <article className="message-row assistant-message">
      <div className="message-avatar assistant"><Icon name="spark" size={17} /></div>
      <div className="message-body">
        <div className="assistant-label">知屿助手</div>
        {message.content ? (
          <div className="answer-content">{message.content}</div>
        ) : message.status === 'streaming' ? (
          <div className="thinking-line"><span /><span /><span />正在检索并组织答案</div>
        ) : (
          <div className="answer-error">回答生成失败，请稍后重试。</div>
        )}
        {(message.citations?.length ?? 0) > 0 && (
          <div className="citation-chips">
            {message.citations!.map((citation, index) => (
              <button key={citation.id || `${citation.document_id}-${citation.page}-${index}`} onClick={() => onCitation(citation)}>
                <span>{citation.index || index + 1}</span>
                {citation.document_name}{citation.page ? ` · P${citation.page}` : ''}
              </button>
            ))}
          </div>
        )}
        {message.status !== 'streaming' && message.content && (
          <div className="message-actions">
            <button type="button" onClick={copyAnswer} aria-live="polite">
              <Icon name={copyState === 'copied' ? 'check' : copyState === 'failed' ? 'alert' : 'copy'} size={14} />
              {copyState === 'copied' ? '已复制' : copyState === 'failed' ? '复制失败' : '复制'}
            </button>
          </div>
        )}
      </div>
    </article>
  )
}

function InspectorPanel({
  message,
  tab,
  setTab,
  selectedCitation,
  setSelectedCitation,
  onClose,
}: {
  message?: ChatMessage
  tab: 'sources' | 'trace'
  setTab: (tab: 'sources' | 'trace') => void
  selectedCitation?: Citation
  setSelectedCitation: (citation?: Citation) => void
  onClose: () => void
}) {
  const citations = message?.citations || []
  const steps = message?.agent_steps || []
  const [openingSource, setOpeningSource] = useState(false)
  const [sourceError, setSourceError] = useState('')

  useEffect(() => {
    setSourceError('')
    setOpeningSource(false)
  }, [selectedCitation])

  async function openSource() {
    if (!selectedCitation?.document_id || openingSource) return

    const target = window.open('about:blank', '_blank')
    if (!target) {
      setSourceError('浏览器阻止了新窗口，请允许弹出窗口后重试。')
      return
    }
    target.opener = null
    target.document.title = '正在打开原文…'
    target.document.body.textContent = '正在加载原文…'
    setOpeningSource(true)
    setSourceError('')

    try {
      const blob = await fetchDocumentFile(selectedCitation.document_id)
      const objectUrl = URL.createObjectURL(blob)
      const pageHash = selectedCitation.page && blob.type === 'application/pdf'
        ? `#page=${selectedCitation.page}`
        : ''
      target.location.replace(`${objectUrl}${pageHash}`)
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 120_000)
    } catch (reason) {
      target.close()
      setSourceError(reason instanceof Error ? reason.message : '无法打开原文')
    } finally {
      setOpeningSource(false)
    }
  }

  return (
    <aside className="inspector-panel">
      <div className="inspector-header">
        <div className="inspector-tabs">
          <button className={tab === 'sources' ? 'active' : ''} onClick={() => setTab('sources')}>
            引用 <span>{citations.length}</span>
          </button>
          <button className={tab === 'trace' ? 'active' : ''} onClick={() => setTab('trace')}>
            Agent 轨迹 <span>{steps.length}</span>
          </button>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="关闭详情"><Icon name="close" size={18} /></button>
      </div>

      {tab === 'sources' && (
        <div className="inspector-content">
          {selectedCitation ? (
            <div className="citation-detail">
              <button className="back-link" onClick={() => setSelectedCitation(undefined)}>← 返回全部引用</button>
              <div className="source-file-icon"><Icon name="file" size={22} /></div>
              <h3>{selectedCitation.document_name}</h3>
              <div className="source-meta">
                {selectedCitation.page && <span>第 {selectedCitation.page} 页</span>}
                {selectedCitation.score !== undefined && <span>相关度 {relevancePercent(selectedCitation.score)}%</span>}
              </div>
              <div className="quoted-content"><Icon name="quote" size={18} /><p>{selectedCitation.content}</p></div>
              {selectedCitation.document_id && (
                <button type="button" className="open-source-link" onClick={() => void openSource()} disabled={openingSource}>
                  {openingSource ? <><Icon name="loader" size={14} className="spin" />正在打开</> : <>打开原文 <Icon name="external" size={14} /></>}
                </button>
              )}
              {sourceError && <p className="source-open-error" role="alert">{sourceError}</p>}
            </div>
          ) : citations.length ? (
            <>
              <div className="inspector-intro"><strong>答案依据</strong><p>以下内容由知识库检索得到，点击可查看原文片段。</p></div>
              <div className="source-list">
                {citations.map((citation, index) => (
                  <button key={citation.id || `${index}-${citation.document_name}`} onClick={() => setSelectedCitation(citation)}>
                    <div className="source-number">{citation.index || index + 1}</div>
                    <div className="source-card-body">
                      <div><strong>{citation.document_name}</strong>{citation.page && <span>P{citation.page}</span>}</div>
                      <p>{citation.content}</p>
                      {citation.score !== undefined && (
                        <div className="relevance"><span style={{ width: `${Math.max(4, relevancePercent(citation.score))}%` }} /><small>{relevancePercent(citation.score)}% 匹配</small></div>
                      )}
                    </div>
                    <Icon name="chevron" size={15} />
                  </button>
                ))}
              </div>
            </>
          ) : (
            <EmptyBlock icon="quote" title="暂无引用" body="完成一次知识库问答后，检索到的原文证据会显示在这里。" />
          )}
        </div>
      )}

      {tab === 'trace' && (
        <div className="inspector-content">
          {steps.length ? (
            <>
              <div className="inspector-intro"><strong>执行过程</strong><p>展示本次回答的检索、分析与生成步骤。</p></div>
              <div className="agent-timeline">
                {steps.map((step, index) => (
                  <div className={`agent-step step-${step.status}`} key={step.id || `${step.name}-${index}`}>
                    <div className="step-rail">
                      <span>{step.status === 'completed' ? <Icon name="check" size={12} /> : step.status === 'failed' ? <Icon name="close" size={12} /> : <span className="step-dot" />}</span>
                    </div>
                    <div className="step-copy">
                      <div><strong>{step.name}</strong>{step.duration_ms !== undefined && <small>{step.duration_ms} ms</small>}</div>
                      {step.detail && <p>{step.detail}</p>}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : message?.status === 'streaming' ? (
            <div className="trace-running"><span className="radar"><i /></span><strong>Agent 正在执行</strong><p>步骤事件到达后会实时显示。</p></div>
          ) : (
            <EmptyBlock icon="route" title="暂无执行轨迹" body="开启 Agent 模式并发起问题后，可在这里查看完整工作流。" />
          )}
        </div>
      )}
    </aside>
  )
}

function ChatView({
  knowledgeBases,
  selectedKnowledgeBases,
  setSelectedKnowledgeBases,
  messages,
  input,
  setInput,
  agentMode,
  setAgentMode,
  sending,
  onSend,
  onStop,
  showInspector,
  setShowInspector,
  inspectorTab,
  setInspectorTab,
  selectedCitation,
  setSelectedCitation,
  onNavigateKnowledge,
}: {
  knowledgeBases: KnowledgeBase[]
  selectedKnowledgeBases: string[]
  setSelectedKnowledgeBases: (ids: string[]) => void
  messages: ChatMessage[]
  input: string
  setInput: (input: string) => void
  agentMode: boolean
  setAgentMode: (value: boolean) => void
  sending: boolean
  onSend: () => void
  onStop: () => void
  showInspector: boolean
  setShowInspector: (value: boolean) => void
  inspectorTab: 'sources' | 'trace'
  setInspectorTab: (value: 'sources' | 'trace') => void
  selectedCitation?: Citation
  setSelectedCitation: (citation?: Citation) => void
  onNavigateKnowledge: () => void
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant')

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  function openCitation(citation: Citation) {
    setSelectedCitation(citation)
    setInspectorTab('sources')
    setShowInspector(true)
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    if (sending) return
    onSend()
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing) return
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (!sending) onSend()
    }
  }

  return (
    <div className={`chat-workspace ${showInspector ? '' : 'inspector-closed'}`}>
      <main className="chat-main">
        <header className="chat-header">
          <div>
            <span className="eyebrow">企业知识问答</span>
            <h1>{messages.length ? '当前对话' : '开始一次知识探索'}</h1>
          </div>
          <div className="chat-header-actions">
            <KnowledgeSelector knowledgeBases={knowledgeBases} selected={selectedKnowledgeBases} onChange={setSelectedKnowledgeBases} />
            {!showInspector && (
              <button className="secondary-button square" onClick={() => setShowInspector(true)} aria-label="打开引用详情"><Icon name="panel" size={17} /></button>
            )}
          </div>
        </header>

        {knowledgeBases.length === 0 && (
          <div className="context-banner">
            <Icon name="alert" size={17} />
            <span><strong>还没有可用知识库</strong> 上传企业 PDF 并完成向量化后即可开始有据可查的问答。</span>
            <button onClick={onNavigateKnowledge}>前往知识库 <Icon name="arrow" size={14} /></button>
          </div>
        )}

        <div className="message-scroll" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <div className="welcome-orbit"><span><Icon name="spark" size={28} /></span></div>
              <span className="welcome-kicker">RAG KNOWLEDGE ASSISTANT</span>
              <h2>把散落的文档，<br />变成随时可问的答案。</h2>
              <p>答案将基于已选企业知识库生成，并附上可回溯的原文引用。</p>
              <div className="welcome-trust" aria-label="系统能力">
                <span><i className="trust-dot green" />知识库约束</span>
                <span><i className="trust-dot orange" />原文可追溯</span>
                <span><i className="trust-dot blue" />智能模型降级</span>
              </div>
              <div className="prompt-starters">
                {[
                  ['提炼要点', '总结知识库中最重要的政策与流程'],
                  ['核对条款', '查找相关制度条款并列出引用来源'],
                  ['跨文档分析', '对比不同文档中的规定与差异'],
                ].map(([label, prompt]) => (
                  <button key={label} onClick={() => setInput(prompt)}>
                    <span>{label}</span><p>{prompt}</p><Icon name="arrow" size={15} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => <MessageBubble key={message.id} message={message} onCitation={openCitation} />)}
            </div>
          )}
        </div>

        <div className="composer-zone">
          <form className={`composer ${sending ? 'is-sending' : ''}`} onSubmit={submit}>
            <textarea
              rows={1}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={knowledgeBases.length ? '向企业知识库提问…' : '请先创建并上传知识库…'}
              disabled={knowledgeBases.length === 0}
            />
            <div className="composer-toolbar">
              <label className="agent-toggle" title="启用多步骤检索和分析">
                <input type="checkbox" checked={agentMode} onChange={(event) => setAgentMode(event.target.checked)} />
                <span className="toggle-track"><i /></span>
                <Icon name="route" size={15} />
                Agent 模式
              </label>
              <span className="composer-hint">Enter 发送 · Shift + Enter 换行</span>
              {sending ? (
                <button type="button" className="send-button stop" onClick={onStop} aria-label="停止生成"><Icon name="stop" size={17} /></button>
              ) : (
                <button type="submit" className="send-button" disabled={!input.trim() || selectedKnowledgeBases.length === 0} aria-label="发送消息"><Icon name="send" size={17} /></button>
              )}
            </div>
          </form>
          <p className="answer-disclaimer">AI 可能会出错，请通过右侧原文引用核对关键信息。</p>
        </div>
      </main>

      {showInspector && (
        <InspectorPanel
          message={lastAssistant}
          tab={inspectorTab}
          setTab={setInspectorTab}
          selectedCitation={selectedCitation}
          setSelectedCitation={setSelectedCitation}
          onClose={() => setShowInspector(false)}
        />
      )}
    </div>
  )
}

function UploadPanel({
  knowledgeBases,
  uploadKnowledgeBase,
  setUploadKnowledgeBase,
  uploadItems,
  setUploadItems,
  onUpload,
  uploading,
}: {
  knowledgeBases: KnowledgeBase[]
  uploadKnowledgeBase: string
  setUploadKnowledgeBase: (id: string) => void
  uploadItems: UploadItem[]
  setUploadItems: React.Dispatch<React.SetStateAction<UploadItem[]>>
  onUpload: () => void
  uploading: boolean
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function acceptFiles(files: File[]) {
    const next: UploadItem[] = []
    for (const file of files) {
      const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
      const tooLarge = file.size > 50 * 1024 * 1024
      next.push({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        file,
        progress: 0,
        status: !isPdf || tooLarge ? 'failed' : 'queued',
        error: !isPdf ? '仅支持 PDF 文件' : tooLarge ? '文件不能超过 50 MB' : undefined,
      })
    }
    setUploadItems((current) => {
      const existing = new Set(current.map((item) => item.id))
      return [...current, ...next.filter((item) => !existing.has(item.id))]
    })
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    acceptFiles(Array.from(event.dataTransfer.files))
  }

  return (
    <section className="upload-panel card">
      <div className="card-heading">
        <div><span className="section-kicker">INGEST</span><h2>添加企业文档</h2></div>
        <select value={uploadKnowledgeBase} onChange={(event) => setUploadKnowledgeBase(event.target.value)} aria-label="目标知识库">
          <option value="">选择目标知识库</option>
          {knowledgeBases.map((item) => <option key={item.id} value={idString(item.id)}>{item.name}</option>)}
        </select>
      </div>
      <div
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            acceptFiles(Array.from(event.target.files || []))
            event.target.value = ''
          }}
        />
        <span className="upload-icon"><Icon name="upload" size={22} /></span>
        <div><strong>拖入 PDF，或点击选择</strong><p>支持批量上传，单个文件不超过 50 MB</p></div>
      </div>

      {uploadItems.length > 0 && (
        <div className="upload-queue">
          {uploadItems.map((item) => (
            <div className="upload-row" key={item.id}>
              <span className="pdf-icon">PDF</span>
              <div className="upload-copy">
                <div><strong>{item.file.name}</strong><small>{formatBytes(item.file.size)}</small></div>
                {item.status === 'uploading' ? (
                  <div className="progress"><span style={{ width: `${item.progress}%` }} /></div>
                ) : item.error ? <p className="row-error">{item.error}</p> : <small>{STATUS_LABELS[item.status]}</small>}
              </div>
              {!uploading && (
                <button className="icon-button" onClick={() => setUploadItems((current) => current.filter((entry) => entry.id !== item.id))} aria-label="移除文件"><Icon name="close" size={16} /></button>
              )}
            </div>
          ))}
          <div className="upload-actions">
            <button className="text-button" disabled={uploading} onClick={() => setUploadItems([])}>清空</button>
            <button className="primary-button" disabled={uploading || !uploadKnowledgeBase || !uploadItems.some((item) => item.status === 'queued')} onClick={onUpload}>
              {uploading ? <><Icon name="loader" className="spin" size={15} />正在上传</> : <><Icon name="upload" size={15} />开始入库</>}
            </button>
          </div>
        </div>
      )}
    </section>
  )
}

function KnowledgeView({
  knowledgeBases,
  documents,
  onRefresh,
  loading,
  onCreateKnowledgeBase,
  onDeleteDocument,
  uploadKnowledgeBase,
  setUploadKnowledgeBase,
  uploadItems,
  setUploadItems,
  onUpload,
  uploading,
}: {
  knowledgeBases: KnowledgeBase[]
  documents: KnowledgeDocument[]
  onRefresh: () => void
  loading: boolean
  onCreateKnowledgeBase: () => void
  onDeleteDocument: (document: KnowledgeDocument) => void
  uploadKnowledgeBase: string
  setUploadKnowledgeBase: (id: string) => void
  uploadItems: UploadItem[]
  setUploadItems: React.Dispatch<React.SetStateAction<UploadItem[]>>
  onUpload: () => void
  uploading: boolean
}) {
  const [filterKb, setFilterKb] = useState('all')
  const [query, setQuery] = useState('')
  const filtered = documents.filter((document) => {
    const matchesKb = filterKb === 'all' || idString(document.knowledge_base_id) === filterKb
    const name = document.name || document.filename || ''
    return matchesKb && name.toLowerCase().includes(query.trim().toLowerCase())
  })
  const readyCount = documents.filter((document) => document.status === 'ready').length
  const processingCount = documents.filter((document) => !['ready', 'failed'].includes(document.status)).length
  const totalChunks = documents.reduce((sum, document) => sum + (document.chunk_count || 0), 0)

  return (
    <main className="knowledge-page">
      <header className="page-header">
        <div><span className="eyebrow">KNOWLEDGE OPERATIONS</span><h1>知识库</h1><p>管理企业文档，从解析到向量索引全程可见。</p></div>
        <div className="page-actions">
          <button className="secondary-button" onClick={onRefresh}><Icon name="refresh" size={15} className={loading ? 'spin' : ''} />刷新</button>
          <button className="primary-button" onClick={onCreateKnowledgeBase}><Icon name="plus" size={16} />新建知识库</button>
        </div>
      </header>

      <section className="metric-grid">
        <div className="metric-card"><span className="metric-icon dark"><Icon name="database" /></span><div><small>知识库</small><strong>{knowledgeBases.length}</strong><p>独立检索空间</p></div></div>
        <div className="metric-card"><span className="metric-icon orange"><Icon name="file" /></span><div><small>已就绪文档</small><strong>{readyCount}</strong><p>共 {documents.length} 份</p></div></div>
        <div className="metric-card"><span className="metric-icon green"><Icon name="layers" /></span><div><small>向量片段</small><strong>{totalChunks.toLocaleString()}</strong><p>可参与召回</p></div></div>
        <div className="metric-card"><span className="metric-icon blue"><Icon name="clock" /></span><div><small>处理中</small><strong>{processingCount}</strong><p>后台任务</p></div></div>
      </section>

      <UploadPanel
        knowledgeBases={knowledgeBases}
        uploadKnowledgeBase={uploadKnowledgeBase}
        setUploadKnowledgeBase={setUploadKnowledgeBase}
        uploadItems={uploadItems}
        setUploadItems={setUploadItems}
        onUpload={onUpload}
        uploading={uploading}
      />

      <section className="document-card card">
        <div className="document-toolbar">
          <div><span className="section-kicker">DOCUMENTS</span><h2>文档索引</h2></div>
          <div className="table-filters">
            <label className="search-field"><Icon name="search" size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文件名" /></label>
            <select value={filterKb} onChange={(event) => setFilterKb(event.target.value)}>
              <option value="all">全部知识库</option>
              {knowledgeBases.map((item) => <option key={item.id} value={idString(item.id)}>{item.name}</option>)}
            </select>
          </div>
        </div>

        {loading && documents.length === 0 ? (
          <div className="table-loading"><Icon name="loader" className="spin" />正在读取文档索引…</div>
        ) : filtered.length === 0 ? (
          <EmptyBlock icon="file" title={documents.length ? '没有匹配的文档' : '知识库还是空的'} body={documents.length ? '尝试清除筛选条件。' : '在上方上传 PDF，系统会自动完成解析、切分和向量化。'} />
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>文档</th><th>知识库</th><th>处理状态</th><th>页数 / 片段</th><th>更新时间</th><th aria-label="操作" /></tr></thead>
              <tbody>
                {filtered.map((document) => (
                  <tr key={document.id}>
                    <td><div className="document-name"><span>PDF</span><div><strong>{document.name || document.filename}</strong><small>{formatBytes(document.size)}</small></div></div></td>
                    <td>{document.knowledge_base_name || knowledgeBases.find((item) => idString(item.id) === idString(document.knowledge_base_id))?.name || '—'}</td>
                    <td><StatusBadge status={document.status} />{document.error && <small className="inline-error" title={document.error}>查看错误</small>}</td>
                    <td>{document.page_count ?? '—'} 页 <span className="separator">/</span> {document.chunk_count ?? '—'} 片段</td>
                    <td>{formatDate(document.updated_at || document.created_at)}</td>
                    <td><button className="icon-button danger" onClick={() => onDeleteDocument(document)} aria-label={`删除 ${document.name || document.filename}`}><Icon name="trash" size={16} /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  )
}

function CreateKnowledgeBaseModal({ onClose, onCreated }: { onClose: () => void; onCreated: (item: KnowledgeBase) => void }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!name.trim()) return
    setSubmitting(true)
    setError('')
    try {
      const item = await createKnowledgeBase({ name: name.trim(), description: description.trim() || undefined })
      onCreated(item)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '创建失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <form className="modal" role="dialog" aria-modal="true" aria-labelledby="create-kb-title" onSubmit={submit}>
        <div className="modal-header"><div><span className="section-kicker">NEW SPACE</span><h2 id="create-kb-title">新建知识库</h2></div><button type="button" className="icon-button" onClick={onClose} aria-label="关闭新建知识库窗口"><Icon name="close" /></button></div>
        <p className="modal-intro">知识库是彼此隔离的文档检索空间，可按部门、项目或业务主题划分。</p>
        <label className="field"><span>名称</span><input autoFocus maxLength={64} value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：人力资源制度" /></label>
        <label className="field"><span>描述 <small>选填</small></span><textarea rows={3} maxLength={240} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="说明这个知识库包含什么内容" /></label>
        {error && <div className="form-error"><Icon name="alert" size={15} />{error}</div>}
        <div className="modal-actions"><button type="button" className="secondary-button" onClick={onClose}>取消</button><button className="primary-button" disabled={!name.trim() || submitting}>{submitting ? '正在创建…' : '创建知识库'}</button></div>
      </form>
    </div>
  )
}

function useDialogFocus(
  dialogRef: RefObject<HTMLElement | null>,
  initialFocusRef: RefObject<HTMLElement | null>,
  onClose: () => void,
  closeEnabled = true,
) {
  const closeRef = useRef(onClose)
  const closeEnabledRef = useRef(closeEnabled)
  closeRef.current = onClose
  closeEnabledRef.current = closeEnabled

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    initialFocusRef.current?.focus()

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape' && closeEnabledRef.current) {
        event.preventDefault()
        closeRef.current()
        return
      }
      if (event.key !== 'Tab' || !dialogRef.current) return

      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )).filter((element) => element.offsetParent !== null && element.getAttribute('aria-hidden') !== 'true')

      if (focusable.length === 0) {
        event.preventDefault()
        dialogRef.current.focus()
        return
      }

      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (event.shiftKey && (active === first || !dialogRef.current.contains(active))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (active === last || !dialogRef.current.contains(active))) {
        event.preventDefault()
        first.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
      if (previouslyFocused?.isConnected) previouslyFocused.focus()
    }
  }, [dialogRef, initialFocusRef])
}

function SessionHistoryPanel({
  sessions,
  activeSessionId,
  deletingSessionId,
  sending,
  onOpenSession,
  onRequestDelete,
  onClose,
}: {
  sessions: ChatSession[]
  activeSessionId?: string
  deletingSessionId?: string
  sending: boolean
  onOpenSession: (session: ChatSession) => void
  onRequestDelete: (session: ChatSession) => void
  onClose: () => void
}) {
  const panelRef = useRef<HTMLElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  useDialogFocus(panelRef, closeButtonRef, onClose)

  return (
    <div className="history-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <aside ref={panelRef} id="history-panel" className="history-panel" role="dialog" aria-modal="true" aria-labelledby="history-panel-title" tabIndex={-1}>
        <div className="history-panel-header">
          <div>
            <span className="section-kicker">CONVERSATIONS</span>
            <h2 id="history-panel-title">最近对话</h2>
          </div>
          <button ref={closeButtonRef} type="button" className="icon-button" onClick={onClose} aria-label="关闭最近对话">
            <Icon name="close" size={18} />
          </button>
        </div>
        <p className="history-panel-intro">
          {sessions.length ? `保留最近 ${Math.min(sessions.length, 12)} 条会话，点击即可继续。` : '完成首次问答后，对话会出现在这里。'}
        </p>
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          deletingSessionId={deletingSessionId}
          sending={sending}
          onOpenSession={onOpenSession}
          onRequestDelete={onRequestDelete}
        />
      </aside>
    </div>
  )
}

function DeleteSessionModal({
  session,
  deleting,
  error,
  onClose,
  onConfirm,
}: {
  session: ChatSession
  deleting: boolean
  error: string
  onClose: () => void
  onConfirm: () => void
}) {
  const title = session.title || '未命名对话'
  const dialogRef = useRef<HTMLDivElement>(null)
  const cancelButtonRef = useRef<HTMLButtonElement>(null)
  useDialogFocus(dialogRef, cancelButtonRef, onClose, !deleting)

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (!deleting && event.target === event.currentTarget) onClose() }}>
      <div ref={dialogRef} className="modal delete-session-modal" role="alertdialog" aria-modal="true" aria-labelledby="delete-session-title" aria-describedby="delete-session-description" tabIndex={-1}>
        <div className="danger-orb"><Icon name="trash" size={22} /></div>
        <div className="modal-header">
          <div><span className="section-kicker">DELETE CONVERSATION</span><h2 id="delete-session-title">删除这条对话？</h2></div>
          <button type="button" className="icon-button" onClick={onClose} disabled={deleting} aria-label="关闭删除确认窗口"><Icon name="close" /></button>
        </div>
        <p className="modal-intro" id="delete-session-description">“{title}”及其中的所有消息将被永久删除，此操作无法撤销。</p>
        {error && <div className="form-error"><Icon name="alert" size={15} />{error}</div>}
        <div className="modal-actions">
          <button ref={cancelButtonRef} type="button" className="secondary-button" onClick={onClose} disabled={deleting}>取消</button>
          <button type="button" className="danger-button" onClick={onConfirm} disabled={deleting}>
            {deleting ? <><Icon name="loader" size={15} className="spin" />正在删除…</> : <><Icon name="trash" size={15} />确认删除</>}
          </button>
        </div>
      </div>
    </div>
  )
}

function Toast({ message, kind, onClose }: { message: string; kind: 'success' | 'error'; onClose: () => void }) {
  useEffect(() => {
    const timeout = window.setTimeout(onClose, 4200)
    return () => window.clearTimeout(timeout)
  }, [onClose])
  return <div className={`toast toast-${kind}`} role={kind === 'error' ? 'alert' : 'status'} aria-live={kind === 'error' ? 'assertive' : 'polite'}><Icon name={kind === 'success' ? 'check' : 'alert'} size={17} /><span>{message}</span><button type="button" onClick={onClose} aria-label="关闭通知"><Icon name="close" size={14} /></button></div>
}

export default function App() {
  const [view, setView] = useState<AppView>('chat')
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string>()
  const [selectedKnowledgeBases, setSelectedKnowledgeBases] = useState<string[]>([])
  const [uploadKnowledgeBase, setUploadKnowledgeBase] = useState('')
  const [uploadItems, setUploadItems] = useState<UploadItem[]>([])
  const [input, setInput] = useState('')
  const [agentMode, setAgentMode] = useState(true)
  const [sending, setSending] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [serviceHealth, setServiceHealth] = useState<ServiceHealth | 'checking'>('checking')
  const [showInspector, setShowInspector] = useState(() => !window.matchMedia('(max-width: 820px)').matches)
  const [inspectorTab, setInspectorTab] = useState<'sources' | 'trace'>('sources')
  const [selectedCitation, setSelectedCitation] = useState<Citation>()
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showSessionHistory, setShowSessionHistory] = useState(false)
  const [sessionToDelete, setSessionToDelete] = useState<ChatSession>()
  const [deletingSessionId, setDeletingSessionId] = useState<string>()
  const [deleteSessionError, setDeleteSessionError] = useState('')
  const [toast, setToast] = useState<{ message: string; kind: 'success' | 'error' }>()
  const abortRef = useRef<AbortController | undefined>(undefined)
  const sessionRequestRef = useRef(0)
  const sessionListRequestRef = useRef(0)

  const refreshAll = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    const sessionListRequestId = ++sessionListRequestRef.current
    const [healthResult, kbResult, docResult, sessionResult] = await Promise.allSettled([
      getHealth(),
      listKnowledgeBases(),
      listDocuments(),
      listChatSessions(),
    ])

    setServiceHealth(healthResult.status === 'fulfilled' ? healthResult.value : 'offline')
    if (kbResult.status === 'fulfilled') setKnowledgeBases(kbResult.value)
    if (docResult.status === 'fulfilled') setDocuments(docResult.value)
    if (sessionResult.status === 'fulfilled' && sessionListRequestRef.current === sessionListRequestId) {
      setSessions(sessionResult.value)
    }
    if (kbResult.status === 'rejected' && docResult.status === 'rejected' && !quiet) {
      setToast({ message: '无法连接到 API，请确认后端服务已启动', kind: 'error' })
    }
    setLoading(false)
  }, [])

  useEffect(() => { void refreshAll() }, [refreshAll])

  useEffect(() => {
    const narrowScreen = window.matchMedia('(max-width: 820px)')
    function closeInspectorOnNarrowScreen(event: MediaQueryListEvent) {
      if (event.matches) setShowInspector(false)
    }
    narrowScreen.addEventListener('change', closeInspectorOnNarrowScreen)
    return () => narrowScreen.removeEventListener('change', closeInspectorOnNarrowScreen)
  }, [])

  useEffect(() => {
    const validIds = new Set(knowledgeBases.map((item) => idString(item.id)))
    const firstId = knowledgeBases.length ? idString(knowledgeBases[0].id) : ''
    setSelectedKnowledgeBases((current) => {
      if (!current.length) return firstId ? [firstId] : []
      const valid = current.filter((id) => validIds.has(id))
      if (valid.length === current.length) return current
      return valid.length ? valid : firstId ? [firstId] : []
    })
    setUploadKnowledgeBase((current) => validIds.has(current) ? current : firstId)
  }, [knowledgeBases])

  useEffect(() => {
    const hasProcessing = documents.some((document) => !['ready', 'failed'].includes(document.status))
    if (!hasProcessing) return
    const timer = window.setInterval(() => {
      void listDocuments().then(setDocuments).catch(() => undefined)
    }, 4000)
    return () => window.clearInterval(timer)
  }, [documents])

  const updateAssistant = useCallback((id: string, updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((current) => current.map((message) => message.id === id ? updater(message) : message))
  }, [])

  async function handleSend() {
    const question = input.trim()
    if (!question || sending || !selectedKnowledgeBases.length) return

    const timestamp = new Date().toISOString()
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: question, created_at: timestamp, status: 'completed' }
    const assistantId = crypto.randomUUID()
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: '', created_at: timestamp, citations: [], agent_steps: [], status: 'streaming' }
    setMessages((current) => [...current, userMessage, assistantMessage])
    setInput('')
    setSending(true)
    setSelectedCitation(undefined)
    setInspectorTab(agentMode ? 'trace' : 'sources')
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const result = await sendChatMessage(
        {
          question,
          knowledge_base_ids: selectedKnowledgeBases,
          session_id: activeSessionId,
          agent_mode: agentMode,
        },
        {
          onAnswerDelta: (delta) => updateAssistant(assistantId, (message) => ({ ...message, content: message.content + delta })),
          onCitation: (citation) => updateAssistant(assistantId, (message) => {
            const exists = message.citations?.some((item) => item.id && citation.id
              ? item.id === citation.id
              : item.document_id === citation.document_id && item.page === citation.page && item.content === citation.content)
            return exists ? message : { ...message, citations: [...(message.citations || []), citation] }
          }),
          onAgentStep: (step) => updateAssistant(assistantId, (message) => {
            const steps = [...(message.agent_steps || [])]
            const match = steps.findIndex((item) => (item.id && step.id && item.id === step.id) || item.name === step.name)
            if (match >= 0) steps[match] = { ...steps[match], ...step }
            else steps.push(step)
            return { ...message, agent_steps: steps }
          }),
        },
        controller.signal,
      )
      updateAssistant(assistantId, (message) => ({ ...message, status: 'completed', run_id: result.run_id }))
      if (result.session_id) setActiveSessionId(idString(result.session_id))
      const sessionListRequestId = ++sessionListRequestRef.current
      void listChatSessions().then((items) => {
        if (sessionListRequestRef.current === sessionListRequestId) setSessions(items)
      }).catch(() => undefined)
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') {
        updateAssistant(assistantId, (message) => ({ ...message, status: 'completed', content: message.content || '已停止生成。' }))
      } else {
        const message = reason instanceof ApiError ? reason.message : '问答服务暂时不可用'
        updateAssistant(assistantId, (item) => ({ ...item, status: 'failed', content: item.content || message }))
        setToast({ message, kind: 'error' })
      }
    } finally {
      setSending(false)
      if (abortRef.current === controller) abortRef.current = undefined
    }
  }

  function newChat() {
    sessionRequestRef.current += 1
    abortRef.current?.abort()
    setMessages([])
    setActiveSessionId(undefined)
    setSelectedCitation(undefined)
    setShowSessionHistory(false)
    setView('chat')
  }

  async function openSession(session: ChatSession) {
    const requestId = ++sessionRequestRef.current
    abortRef.current?.abort()
    setShowSessionHistory(false)
    setView('chat')
    setActiveSessionId(idString(session.id))
    setSelectedCitation(undefined)
    if (session.messages !== undefined) {
      setMessages(session.messages)
      return
    }
    setMessages([])
    try {
      const detail = await getChatSession(idString(session.id))
      if (sessionRequestRef.current === requestId) setMessages(detail.messages || [])
    } catch (reason) {
      if (sessionRequestRef.current === requestId) {
        setToast({ message: reason instanceof Error ? reason.message : '无法读取历史对话', kind: 'error' })
      }
    }
  }

  function requestDeleteSession(session: ChatSession) {
    const sessionId = idString(session.id)
    if (sending && idString(activeSessionId) === sessionId) {
      setToast({ message: '请先停止当前回答，再删除这条对话', kind: 'error' })
      return
    }
    if (deletingSessionId) return
    setShowSessionHistory(false)
    setDeleteSessionError('')
    setSessionToDelete(session)
  }

  function removeSessionFromView(sessionId: string, alreadyMissing = false) {
    sessionListRequestRef.current += 1
    setSessions((current) => current.filter((item) => idString(item.id) !== sessionId))
    if (idString(activeSessionId) === sessionId) newChat()
    setSessionToDelete(undefined)
    setDeleteSessionError('')
    setToast({
      message: alreadyMissing ? '该对话已不存在，列表已同步' : '最近对话已删除',
      kind: 'success',
    })
  }

  async function confirmDeleteSession() {
    if (!sessionToDelete || deletingSessionId) return
    const sessionId = idString(sessionToDelete.id)
    setDeletingSessionId(sessionId)
    setDeleteSessionError('')
    try {
      await deleteChatSession(sessionId)
      removeSessionFromView(sessionId)
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 404) {
        removeSessionFromView(sessionId, true)
      } else {
        const message = reason instanceof Error ? reason.message : '无法删除这条对话'
        setDeleteSessionError(message)
        setToast({ message, kind: 'error' })
      }
    } finally {
      setDeletingSessionId(undefined)
    }
  }

  async function handleUpload() {
    const pending = uploadItems.filter((item) => item.status === 'queued')
    if (!pending.length || !uploadKnowledgeBase) return
    setUploading(true)
    setUploadItems((current) => current.map((item) => pending.some((entry) => entry.id === item.id) ? { ...item, status: 'uploading', progress: 0 } : item))
    try {
      await uploadDocuments(pending.map((item) => item.file), uploadKnowledgeBase, (progress) => {
        setUploadItems((current) => current.map((item) => pending.some((entry) => entry.id === item.id) ? { ...item, progress } : item))
      })
      setUploadItems((current) => current.map((item) => pending.some((entry) => entry.id === item.id) ? { ...item, status: 'parsing', progress: 100 } : item))
      setToast({ message: `${pending.length} 份文档已提交处理`, kind: 'success' })
      await refreshAll(true)
      window.setTimeout(() => setUploadItems((current) => current.filter((item) => !pending.some((entry) => entry.id === item.id))), 1800)
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : '上传失败'
      setUploadItems((current) => current.map((item) => pending.some((entry) => entry.id === item.id) ? { ...item, status: 'failed', error: message } : item))
      setToast({ message, kind: 'error' })
    } finally {
      setUploading(false)
    }
  }

  async function handleDeleteDocument(document: KnowledgeDocument) {
    const name = document.name || document.filename || '该文档'
    if (!window.confirm(`确定删除“${name}”吗？对应的向量索引也会一并移除。`)) return
    try {
      await deleteDocument(idString(document.id))
      setDocuments((current) => current.filter((item) => idString(item.id) !== idString(document.id)))
      setToast({ message: '文档及其向量索引已删除', kind: 'success' })
    } catch (reason) {
      setToast({ message: reason instanceof Error ? reason.message : '删除失败', kind: 'error' })
    }
  }

  return (
    <div className="app-shell">
      <AppSidebar
        view={view}
        setView={setView}
        sessions={sessions}
        activeSessionId={activeSessionId}
        deletingSessionId={deletingSessionId}
        sending={sending}
        onOpenSession={openSession}
        onRequestDelete={requestDeleteSession}
        onShowHistory={() => setShowSessionHistory(true)}
        historyOpen={showSessionHistory}
        onNewChat={newChat}
      />
      <MobileNav view={view} setView={setView} onShowHistory={() => setShowSessionHistory(true)} historyOpen={showSessionHistory} onNewChat={newChat} />
      <div className="app-stage">
        <div className="connection-state" title={SERVICE_HEALTH_COPY[serviceHealth].title}>
          <span className={serviceHealth} />
          {SERVICE_HEALTH_COPY[serviceHealth].label}
        </div>
        {view === 'chat' ? (
          <ChatView
            knowledgeBases={knowledgeBases}
            selectedKnowledgeBases={selectedKnowledgeBases}
            setSelectedKnowledgeBases={setSelectedKnowledgeBases}
            messages={messages}
            input={input}
            setInput={setInput}
            agentMode={agentMode}
            setAgentMode={setAgentMode}
            sending={sending}
            onSend={handleSend}
            onStop={() => abortRef.current?.abort()}
            showInspector={showInspector}
            setShowInspector={setShowInspector}
            inspectorTab={inspectorTab}
            setInspectorTab={setInspectorTab}
            selectedCitation={selectedCitation}
            setSelectedCitation={setSelectedCitation}
            onNavigateKnowledge={() => setView('knowledge')}
          />
        ) : (
          <KnowledgeView
            knowledgeBases={knowledgeBases}
            documents={documents}
            onRefresh={() => void refreshAll()}
            loading={loading}
            onCreateKnowledgeBase={() => setShowCreateModal(true)}
            onDeleteDocument={handleDeleteDocument}
            uploadKnowledgeBase={uploadKnowledgeBase}
            setUploadKnowledgeBase={setUploadKnowledgeBase}
            uploadItems={uploadItems}
            setUploadItems={setUploadItems}
            onUpload={handleUpload}
            uploading={uploading}
          />
        )}
      </div>

      {showCreateModal && (
        <CreateKnowledgeBaseModal
          onClose={() => setShowCreateModal(false)}
          onCreated={(item) => {
            setKnowledgeBases((current) => [...current, item])
            setUploadKnowledgeBase(idString(item.id))
            setShowCreateModal(false)
            setToast({ message: '知识库创建成功', kind: 'success' })
          }}
        />
      )}
      {showSessionHistory && (
        <SessionHistoryPanel
          sessions={sessions}
          activeSessionId={activeSessionId}
          deletingSessionId={deletingSessionId}
          sending={sending}
          onOpenSession={openSession}
          onRequestDelete={requestDeleteSession}
          onClose={() => setShowSessionHistory(false)}
        />
      )}
      {sessionToDelete && (
        <DeleteSessionModal
          session={sessionToDelete}
          deleting={Boolean(deletingSessionId)}
          error={deleteSessionError}
          onClose={() => {
            if (deletingSessionId) return
            setSessionToDelete(undefined)
            setDeleteSessionError('')
          }}
          onConfirm={() => void confirmDeleteSession()}
        />
      )}
      {toast && <Toast message={toast.message} kind={toast.kind} onClose={() => setToast(undefined)} />}
    </div>
  )
}
