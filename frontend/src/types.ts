export type AppView = 'chat' | 'knowledge'

export type DocumentStatus =
  | 'queued'
  | 'uploading'
  | 'processing'
  | 'parsing'
  | 'chunking'
  | 'embedding'
  | 'indexing'
  | 'ready'
  | 'failed'

export interface KnowledgeBase {
  id: string
  name: string
  description?: string
  document_count?: number
  chunk_count?: number
  updated_at?: string
  status?: 'ready' | 'indexing' | 'error'
}

export interface KnowledgeDocument {
  id: string
  name: string
  filename?: string
  knowledge_base_id?: string
  knowledge_base_name?: string
  status: DocumentStatus
  size?: number
  byte_size?: number
  page_count?: number
  chunk_count?: number
  created_at?: string
  updated_at?: string
  error?: string
}

export interface Citation {
  id?: string
  index?: number
  document_id?: string
  document_name: string
  page?: number
  content: string
  excerpt?: string
  score?: number
  metadata?: Record<string, unknown>
}

export interface AgentStep {
  id?: string
  name: string
  detail?: string
  status: 'pending' | 'running' | 'completed' | 'skipped' | 'failed'
  duration_ms?: number
  started_at?: string
  title?: string
  tool?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  citations?: Citation[]
  agent_steps?: AgentStep[]
  run_id?: string
  status?: 'streaming' | 'completed' | 'failed'
}

export interface ChatSession {
  id: string
  title?: string
  created_at?: string
  updated_at?: string
  messages?: ChatMessage[]
}

export interface ChatResponse {
  answer: string
  session_id?: string
  run_id?: string
  citations?: Citation[]
  agent_steps?: AgentStep[]
}

export interface UploadItem {
  id: string
  file: File
  progress: number
  status: DocumentStatus
  error?: string
}
