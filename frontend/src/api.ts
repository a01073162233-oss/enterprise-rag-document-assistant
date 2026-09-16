import type {
  AgentStep,
  ChatResponse,
  ChatSession,
  Citation,
  KnowledgeBase,
  KnowledgeDocument,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
const API_KEY = import.meta.env.VITE_RAG_API_KEY || ''

function authenticationHeaders(): Record<string, string> {
  return API_KEY ? { 'X-API-Key': API_KEY } : {}
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    return payload.detail || payload.message || `请求失败（${response.status}）`
  } catch {
    return `请求失败（${response.status}）`
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
      ...authenticationHeaders(),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    throw new ApiError(await parseError(response), response.status)
  }

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function unwrapList<T>(payload: T[] | { items?: T[]; data?: T[] }): T[] {
  if (Array.isArray(payload)) return payload
  return payload.items || payload.data || []
}

function normalizeDocument(value: KnowledgeDocument | Record<string, unknown>): KnowledgeDocument {
  const item = value as Record<string, unknown>
  return {
    ...item,
    id: String(item.id ?? ''),
    name: String(item.name ?? item.filename ?? '未命名文档'),
    filename: item.filename as string | undefined,
    knowledge_base_id: item.knowledge_base_id as string | undefined,
    knowledge_base_name: item.knowledge_base_name as string | undefined,
    status: String(item.status ?? 'processing') as KnowledgeDocument['status'],
    size: item.size === undefined && item.byte_size === undefined ? undefined : Number(item.size ?? item.byte_size),
    byte_size: item.byte_size === undefined ? undefined : Number(item.byte_size),
    page_count: item.page_count === undefined ? undefined : Number(item.page_count),
    chunk_count: item.chunk_count === undefined ? undefined : Number(item.chunk_count),
    created_at: item.created_at as string | undefined,
    updated_at: (item.updated_at ?? item.processed_at ?? item.created_at) as string | undefined,
    error: item.error as string | undefined,
  }
}

export type ServiceHealth = 'online' | 'degraded' | 'offline'

export async function getHealth(): Promise<ServiceHealth> {
  try {
    const payload = await request<{ status?: string; vector_database?: string }>('/health')
    const degraded = (payload.status !== undefined && payload.status !== 'ok')
      || payload.vector_database === 'unavailable'
    return degraded ? 'degraded' : 'online'
  } catch {
    return 'offline'
  }
}

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const payload = await request<KnowledgeBase[] | { items?: KnowledgeBase[]; data?: KnowledgeBase[] }>(
    '/knowledge-bases',
  )
  return unwrapList(payload)
}

export async function createKnowledgeBase(input: {
  name: string
  description?: string
}): Promise<KnowledgeBase> {
  return request<KnowledgeBase>('/knowledge-bases', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export async function listDocuments(knowledgeBaseId?: string): Promise<KnowledgeDocument[]> {
  const query = knowledgeBaseId ? `?knowledge_base_id=${encodeURIComponent(knowledgeBaseId)}` : ''
  const payload = await request<
    KnowledgeDocument[] | { items?: KnowledgeDocument[]; data?: KnowledgeDocument[] }
  >(`/documents${query}`)
  return unwrapList(payload).map(normalizeDocument)
}

export async function uploadDocuments(
  files: File[],
  knowledgeBaseId: string,
  onProgress?: (progress: number) => void,
): Promise<KnowledgeDocument[]> {
  const form = new FormData()
  files.forEach((file) => form.append('files', file))
  form.append('knowledge_base_id', knowledgeBaseId)

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/documents/upload`)
    if (API_KEY) xhr.setRequestHeader('X-API-Key', API_KEY)
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress?.(Math.round((event.loaded / event.total) * 100))
    })
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const payload = JSON.parse(xhr.responseText)
          resolve(unwrapList<KnowledgeDocument>(payload).map(normalizeDocument))
        } catch {
          resolve([])
        }
        return
      }
      try {
        const payload = JSON.parse(xhr.responseText)
        reject(new ApiError(payload.detail || payload.message || `上传失败（${xhr.status}）`, xhr.status))
      } catch {
        reject(new ApiError(`上传失败（${xhr.status}）`, xhr.status))
      }
    })
    xhr.addEventListener('error', () => reject(new ApiError('无法连接到文档服务', 0)))
    xhr.send(form)
  })
}

export async function deleteDocument(documentId: string): Promise<void> {
  return request<void>(`/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' })
}

export async function fetchDocumentFile(documentId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(documentId)}/file`, {
    headers: authenticationHeaders(),
  })
  if (!response.ok) throw new ApiError(await parseError(response), response.status)
  return response.blob()
}

export async function listChatSessions(): Promise<ChatSession[]> {
  const payload = await request<ChatSession[] | { items?: ChatSession[]; data?: ChatSession[] }>(
    '/chat/sessions',
  )
  return unwrapList(payload)
}

export async function getChatSession(sessionId: string): Promise<ChatSession> {
  const payload = await request<Omit<ChatSession, 'messages'> & { messages?: Array<Record<string, unknown>> }>(
    `/chat/sessions/${encodeURIComponent(sessionId)}`,
  )
  return {
    ...payload,
    messages: (payload.messages || []).map((message) => ({
      id: String(message.id ?? crypto.randomUUID()),
      role: message.role === 'user' ? 'user' : 'assistant',
      content: String(message.content ?? ''),
      created_at: String(message.created_at ?? new Date().toISOString()),
      citations: ((message.citations ?? message.sources ?? []) as unknown[]).map(normalizeCitation),
      agent_steps: ((message.agent_steps ?? message.trace ?? []) as unknown[]).map(normalizeAgentStep),
      run_id: message.run_id as string | undefined,
      status: 'completed',
    })),
  }
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  return request<void>(`/chat/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

interface StreamHandlers {
  onAnswerDelta: (delta: string) => void
  onCitation: (citation: Citation) => void
  onAgentStep: (step: AgentStep) => void
}

function normalizeCitation(value: unknown): Citation {
  const item = (value || {}) as Record<string, unknown>
  return {
    ...item,
    id: item.id as string | undefined,
    index: item.index as number | undefined,
    document_id: (item.document_id ?? item.id) as string | undefined,
    document_name: String(item.document_name ?? item.filename ?? item.source ?? '未知文档'),
    page: item.page === undefined || item.page === null ? undefined : Number(item.page),
    content: String(item.content ?? item.excerpt ?? item.text ?? ''),
    excerpt: item.excerpt as string | undefined,
    score: item.score === undefined || item.score === null ? undefined : Number(item.score),
    metadata: item.metadata as Record<string, unknown> | undefined,
  }
}

function normalizeAgentStep(value: unknown): AgentStep {
  const item = (value || {}) as Record<string, unknown>
  const rawStatus = String(item.status ?? 'completed')
  const status: AgentStep['status'] = rawStatus === 'done' || rawStatus === 'success'
    ? 'completed'
    : rawStatus === 'error'
      ? 'failed'
      : (['pending', 'running', 'completed', 'skipped', 'failed'].includes(rawStatus) ? rawStatus : 'completed') as AgentStep['status']
  return {
    ...item,
    id: item.id as string | undefined,
    name: String(item.name ?? item.title ?? item.tool ?? 'Agent 步骤'),
    title: item.title as string | undefined,
    detail: item.detail as string | undefined,
    status,
    duration_ms: item.duration_ms === undefined ? undefined : Number(item.duration_ms),
    started_at: item.started_at as string | undefined,
    tool: item.tool as string | undefined,
  }
}

function normalizeChatResponse(payload: Record<string, unknown>): ChatResponse {
  const citationItems = (payload.citations ?? payload.sources ?? []) as unknown[]
  const stepItems = (payload.agent_steps ?? payload.trace ?? payload.steps ?? []) as unknown[]
  return {
    answer: String(payload.answer ?? payload.content ?? payload.message ?? ''),
    session_id: (payload.session_id ?? payload.conversation_id) as string | undefined,
    run_id: payload.run_id as string | undefined,
    citations: citationItems.map(normalizeCitation),
    agent_steps: stepItems.map(normalizeAgentStep),
  }
}

function citationsMatch(left: Citation, right: Citation): boolean {
  if (left.id && right.id) return left.id === right.id
  return left.document_id === right.document_id
    && left.page === right.page
    && left.content === right.content
}

function handleStreamEvent(
  raw: string,
  handlers: StreamHandlers,
  aggregate: ChatResponse,
): ChatResponse {
  if (!raw.trim() || raw.trim() === '[DONE]') return aggregate
  try {
    const event = JSON.parse(raw) as Record<string, unknown>
    const type = event.type || event.event
    if (type === 'answer_delta' || type === 'token' || type === 'content') {
      const delta = String(event.delta ?? event.content ?? '')
      aggregate.answer += delta
      handlers.onAnswerDelta(delta)
    } else if (type === 'citation' || type === 'retrieval_hit') {
      const citation = normalizeCitation(event.citation ?? event.data ?? event)
      aggregate.citations = [...(aggregate.citations || []), citation]
      handlers.onCitation(citation)
    } else if (type === 'agent_step' || type === 'step') {
      const step = normalizeAgentStep(event.step ?? event.data ?? event)
      aggregate.agent_steps = [...(aggregate.agent_steps || []), step]
      handlers.onAgentStep(step)
    } else if (type === 'completed' || type === 'done') {
      const completed = event.data && typeof event.data === 'object'
        ? event.data as Record<string, unknown>
        : event
      aggregate.session_id = (completed.session_id ?? event.session_id ?? aggregate.session_id) as string | undefined
      aggregate.run_id = (completed.run_id ?? event.run_id ?? aggregate.run_id) as string | undefined
      if (completed.answer && !aggregate.answer) {
        const answer = String(completed.answer)
        aggregate.answer = answer
        handlers.onAnswerDelta(answer)
      }
      if (completed.citations || completed.sources) {
        const finalCitations = ((completed.citations ?? completed.sources) as unknown[]).map(normalizeCitation)
        for (const citation of finalCitations) {
          if ((aggregate.citations || []).some((item) => citationsMatch(item, citation))) continue
          aggregate.citations = [...(aggregate.citations || []), citation]
          handlers.onCitation(citation)
        }
      }
      if (completed.agent_steps || completed.trace) {
        const finalSteps = ((completed.agent_steps ?? completed.trace) as unknown[]).map(normalizeAgentStep)
        for (const step of finalSteps) {
          const steps = [...(aggregate.agent_steps || [])]
          const index = steps.findIndex((item) => (item.id && step.id && item.id === step.id) || item.name === step.name)
          if (index >= 0) steps[index] = { ...steps[index], ...step }
          else steps.push(step)
          aggregate.agent_steps = steps
          handlers.onAgentStep(step)
        }
      }
    }
  } catch {
    aggregate.answer += raw
    handlers.onAnswerDelta(raw)
  }
  return aggregate
}

export async function sendChatMessage(
  input: {
    question: string
    knowledge_base_ids: string[]
    session_id?: string
    agent_mode: boolean
  },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream, application/json',
      ...authenticationHeaders(),
    },
    body: JSON.stringify(input),
    signal,
  })

  if (!response.ok) throw new ApiError(await parseError(response), response.status)

  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('text/event-stream') || !response.body) {
    const payload = (await response.json()) as Record<string, unknown>
    const normalized = normalizeChatResponse(payload)
    if (normalized.answer) handlers.onAnswerDelta(normalized.answer)
    normalized.citations?.forEach(handlers.onCitation)
    normalized.agent_steps?.forEach(handlers.onAgentStep)
    return normalized
  }

  const aggregate: ChatResponse = { answer: '', citations: [], agent_steps: [] }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split(/\r?\n\r?\n/)
    buffer = blocks.pop() || ''
    for (const block of blocks) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      handleStreamEvent(data, handlers, aggregate)
    }
  }

  buffer += decoder.decode()

  if (buffer.trim()) {
    const data = buffer.replace(/^data:\s?/gm, '').trim()
    handleStreamEvent(data, handlers, aggregate)
  }
  return aggregate
}
