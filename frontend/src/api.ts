const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface DocumentRecord {
  id: string;
  name: string;
  created_at: string;
  metadata: Record<string, string>;
  segment_count: number;
  chunk_count: number;
  chunking_strategy: "fixed" | "recursive";
  chunk_size: number;
  chunk_overlap: number;
}

export interface Health {
  status: string;
  ollama: {
    available: boolean;
    models: string[];
    error: string | null;
  };
  configured_models: {
    embedding: string;
    generation: string;
  };
}

export interface ChatResponse {
  answer: string;
  retrieval_strategy: "bm25" | "dense";
  models: { embedding: string | null; generation: string };
  sources: {
    source_id: string;
    chunk_id: string;
    document_id: string;
    document_name: string;
    page_number: number | null;
    section_heading: string | null;
    source_location: string;
    content: string;
    score: number;
  }[];
  timings_ms: { retrieval: number; generation: number; total: number };
  generation_profile: {
    model: string;
    model_load_ms: number | null;
    prompt_eval_ms: number | null;
    token_generation_ms: number | null;
    ollama_total_ms: number | null;
    prompt_tokens: number | null;
    output_tokens: number | null;
    generation_tokens_per_second: number | null;
    cold_start: boolean | null;
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail ?? message;
    } catch {
      // Keep the HTTP status message when the server response is not JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  documents: () => request<DocumentRecord[]>("/api/documents"),
  upload: (form: FormData) =>
    request<{ document: DocumentRecord }>("/api/documents", {
      method: "POST",
      body: form,
    }),
  remove: (id: string) =>
    request<void>(`/api/documents/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  chat: (payload: {
    query: string;
    retrieval_strategy: "bm25" | "dense";
    top_k: number;
  }) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
