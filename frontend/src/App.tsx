import { FormEvent, useCallback, useEffect, useState } from "react";
import { api, ChatResponse, DocumentRecord, Health } from "./api";

type Message = {
  role: "user" | "assistant";
  text: string;
  response?: ChatResponse;
};

function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [loadError, setLoadError] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [chunking, setChunking] = useState<"fixed" | "recursive">("fixed");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [retrieval, setRetrieval] = useState<"bm25" | "dense">("bm25");
  const [topK, setTopK] = useState(5);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  const refresh = useCallback(async () => {
    try {
      const [documentList, runtime] = await Promise.all([
        api.documents(),
        api.health(),
      ]);
      setDocuments(documentList);
      setHealth(runtime);
      setLoadError("");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Could not load the app.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    const formElement = event.currentTarget;
    setUploading(true);
    setLoadError("");
    const form = new FormData();
    form.set("file", file);
    form.set("chunking_strategy", chunking);
    form.set("chunk_size", String(chunkSize));
    form.set("chunk_overlap", String(chunkOverlap));
    try {
      await api.upload(form);
      setFile(null);
      formElement.reset();
      await refresh();
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Document upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function removeDocument(document: DocumentRecord) {
    if (!window.confirm(`Remove "${document.name}" and its indexed chunks?`)) return;
    try {
      await api.remove(document.id);
      await refresh();
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Document removal failed.");
    }
  }

  async function sendQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = query.trim();
    if (!question || busy) return;
    setMessages((current) => [...current, { role: "user", text: question }]);
    setQuery("");
    setBusy(true);
    try {
      const response = await api.chat({
        query: question,
        retrieval_strategy: retrieval,
        top_k: topK,
      });
      setMessages((current) => [
        ...current,
        { role: "assistant", text: response.answer, response },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: error instanceof Error ? error.message : "The request failed.",
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-600 text-lg font-bold text-white">
              E
            </div>
            <div>
              <p className="text-sm font-bold tracking-wide">ENGINEERING RAG</p>
              <p className="text-xs text-slate-500">Research workspace · Phase 1</p>
            </div>
          </div>
          <div
            className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ${
              health?.ollama.available
                ? "bg-emerald-50 text-emerald-700"
                : "bg-amber-50 text-amber-800"
            }`}
            title={health?.ollama.error ?? "Ollama is available"}
          >
            <span className="h-2 w-2 rounded-full bg-current" />
            Ollama {health?.ollama.available ? "connected" : "not connected"}
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[350px_minmax(0,1fr)]">
        <aside className="space-y-6">
          <section className="panel">
            <div className="mb-5">
              <p className="eyebrow">KNOWLEDGE BASE</p>
              <h1 className="mt-1 text-xl font-bold">Documents</h1>
              <p className="mt-1 text-sm text-slate-500">
                Upload manuals and specifications to index them locally.
              </p>
            </div>
            <form className="space-y-4" onSubmit={uploadDocument}>
              <label className="field-label">
                Document file
                <input
                  className="file-input"
                  type="file"
                  accept=".pdf,.docx,.txt"
                  required
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <label className="field-label">
                Chunking strategy
                <select
                  className="control"
                  value={chunking}
                  onChange={(event) =>
                    setChunking(event.target.value as "fixed" | "recursive")
                  }
                >
                  <option value="fixed">Fixed size</option>
                  <option value="recursive">Recursive</option>
                </select>
              </label>
              <div className="grid grid-cols-2 gap-3">
                <label className="field-label">
                  Chunk size (chars)
                  <input
                    className="control"
                    type="number"
                    min={1}
                    max={10000}
                    value={chunkSize}
                    onChange={(event) => setChunkSize(Number(event.target.value))}
                  />
                </label>
                <label className="field-label">
                  Overlap
                  <input
                    className="control"
                    type="number"
                    min={0}
                    max={9999}
                    value={chunkOverlap}
                    onChange={(event) => setChunkOverlap(Number(event.target.value))}
                  />
                </label>
              </div>
              {chunkOverlap >= chunkSize && (
                <p className="text-xs text-rose-700">
                  Overlap must be smaller than the chunk size.
                </p>
              )}
              <button
                className="primary-button w-full"
                type="submit"
                disabled={!file || uploading || chunkOverlap >= chunkSize}
              >
                {uploading ? "Processing document..." : "Upload and index"}
              </button>
            </form>
          </section>

          <section className="panel">
            <div className="mb-4 flex items-end justify-between">
              <div>
                <p className="eyebrow">INDEX</p>
                <h2 className="mt-1 text-lg font-bold">Your library</h2>
              </div>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
                {documents.length} files
              </span>
            </div>
            {documents.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-500">
                No documents yet. Upload a PDF, DOCX, or TXT file to begin.
              </div>
            ) : (
              <ul className="space-y-3">
                {documents.map((document) => (
                  <li
                    className="rounded-lg border border-slate-200 p-3"
                    key={document.id}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="break-all text-sm font-semibold">{document.name}</p>
                      <button
                        className="text-xs font-medium text-rose-700 hover:text-rose-900"
                        type="button"
                        onClick={() => void removeDocument(document)}
                      >
                        Remove
                      </button>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">
                      {document.chunk_count} chunks · {document.chunking_strategy} ·{" "}
                      {document.chunk_size} chars
                    </p>
                    {document.metadata.title && (
                      <p className="mt-1 truncate text-xs text-slate-500">
                        {document.metadata.title}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
          {health && (
            <section className="panel text-xs text-slate-600">
              <p className="eyebrow">LOCAL MODELS</p>
              <p className="mt-2">
                Embeddings: <span className="font-semibold">{health.configured_models.embedding}</span>
              </p>
              <p className="mt-1">
                Generation: <span className="font-semibold">{health.configured_models.generation}</span>
              </p>
              {!health.ollama.available && (
                <p className="mt-3 leading-relaxed text-amber-800">
                  Start Ollama and pull both models to enable dense retrieval and
                  answer generation.
                </p>
              )}
            </section>
          )}
        </aside>

        <section className="panel flex min-h-[720px] flex-col">
          <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-100 pb-5">
            <div>
              <p className="eyebrow">DOCUMENT ASSISTANT</p>
              <h2 className="mt-1 text-xl font-bold">Engineering chat</h2>
              <p className="mt-1 text-sm text-slate-500">
                Answers are generated from retrieved document evidence.
              </p>
            </div>
            <div className="flex gap-2">
              <label className="field-label min-w-36">
                Retrieval
                <select
                  className="control"
                  value={retrieval}
                  onChange={(event) =>
                    setRetrieval(event.target.value as "bm25" | "dense")
                  }
                >
                  <option value="bm25">BM25</option>
                  <option value="dense">Dense · Ollama</option>
                </select>
              </label>
              <label className="field-label w-24">
                Top K
                <input
                  className="control"
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
            </div>
          </div>

          {loadError && (
            <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
              {loadError}
            </div>
          )}

          <div className="flex-1 space-y-5 overflow-y-auto py-6">
            {messages.length === 0 ? (
              <div className="grid h-full min-h-72 place-items-center text-center">
                <div className="max-w-sm">
                  <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-indigo-50 text-xl text-indigo-700">
                    +
                  </div>
                  <h3 className="mt-4 font-semibold">Ask your documents</h3>
                  <p className="mt-2 text-sm leading-relaxed text-slate-500">
                    Try a specification question, such as “What is the maximum
                    operating pressure?” Sources and page references appear with
                    every answer.
                  </p>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  className={message.role === "user" ? "ml-auto max-w-[85%]" : "mr-auto max-w-[95%]"}
                  key={`${message.role}-${index}`}
                >
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    {message.role === "user" ? "You" : "Engineering RAG"}
                  </p>
                  <div
                    className={
                      message.role === "user"
                        ? "whitespace-pre-wrap rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-3 text-sm leading-relaxed text-white"
                        : "whitespace-pre-wrap rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-3 text-sm leading-relaxed text-slate-800"
                    }
                  >
                    {message.text}
                  </div>
                  {message.response && (
                    <div className="mt-3 space-y-3">
                      <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                        <span className="rounded bg-slate-100 px-2 py-1">
                          Retrieve {message.response.timings_ms.retrieval} ms
                        </span>
                        <span className="rounded bg-slate-100 px-2 py-1">
                          Generate {message.response.timings_ms.generation} ms
                        </span>
                        <span className="rounded bg-slate-100 px-2 py-1">
                          {message.response.retrieval_strategy.toUpperCase()}
                        </span>
                      </div>
                      {message.response.sources.map((source) => (
                        <details
                          className="rounded-xl border border-slate-200 bg-white"
                          key={source.chunk_id}
                        >
                          <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-indigo-800">
                            [{source.source_id}] {source.document_name}
                            {source.page_number ? ` · page ${source.page_number}` : ""}
                            {source.section_heading ? ` · ${source.section_heading}` : ""}
                          </summary>
                          <p className="border-t border-slate-100 px-3 py-3 text-xs leading-relaxed text-slate-600">
                            {source.content}
                          </p>
                        </details>
                      ))}
                    </div>
                  )}
                </article>
              ))
            )}
            {busy && (
              <p className="text-sm text-slate-500" role="status">
                Retrieving evidence and generating an answer...
              </p>
            )}
          </div>

          <form className="border-t border-slate-100 pt-5" onSubmit={sendQuery}>
            <label className="sr-only" htmlFor="question">
              Ask an engineering question
            </label>
            <div className="flex items-end gap-3">
              <textarea
                className="min-h-12 flex-1 resize-y rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                id="question"
                placeholder="Ask a question about your engineering documents..."
                rows={2}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                disabled={busy || documents.length === 0}
              />
              <button className="primary-button h-12" type="submit" disabled={busy || !query.trim() || documents.length === 0}>
                Send
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-400">
              Verify critical engineering values against the cited source.
            </p>
          </form>
        </section>
      </div>
    </main>
  );
}

export default App;
