import axios from 'axios';
import { useMemo, useState } from 'react';

const API = 'http://localhost:8000';

function makeSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [pdf, setPdf] = useState(null);
  const [excel, setExcel] = useState(null);
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(4);
  const [log, setLog] = useState('');
  const [messages, setMessages] = useState([]);
  const [download, setDownload] = useState(null);
  const [asking, setAsking] = useState(false);
  const sessionId = useMemo(makeSessionId, []);

  const append = (text) => setLog(prev => `${prev}\n${text}`.trim());

  async function ingestPdf() {
    if (!pdf) return alert('Select a PDF first');
    const form = new FormData();
    form.append('file', pdf);
    append('Indexing PDF...');
    const res = await axios.post(`${API}/ingest-pdf`, form);
    append(JSON.stringify(res.data, null, 2));
  }

  async function ask() {
    const text = question.trim();
    if (!text) return alert('Enter a question');

    setQuestion('');
    setAsking(true);
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    append('Generating answer...');

    try {
      const res = await axios.post(`${API}/ask`, {
        question: text,
        top_k: Number(topK),
        session_id: sessionId,
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        contexts: res.data.contexts || [],
        piiRedactionLog: res.data.pii_redaction_log || [],
        answerPiiRedactionLog: res.data.answer_pii_redaction_log || [],
      }]);
      append('Answer generated.');
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${detail}` }]);
      append(`Ask failed: ${detail}`);
    } finally {
      setAsking(false);
    }
  }

  async function clearConversation() {
    setMessages([]);
    setQuestion('');
    await axios.delete(`${API}/chat/${sessionId}`).catch(() => null);
    append('Conversation cleared.');
  }

  async function evaluate() {
    if (!excel) return alert('Select an Excel evaluation file first');
    const form = new FormData();
    form.append('file', excel);
    form.append('top_k', topK);
    append('Running RAGAS evaluation...');
    const res = await axios.post(`${API}/evaluate`, form);
    setDownload(`${API}${res.data.download_url}`);
    append(JSON.stringify(res.data, null, 2));
  }

  function onQuestionKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!asking) ask();
    }
  }

  return (
    <div className="container">
      <h1>PDF RAG + RAGAS Evaluation</h1>
      <p>Upload a PDF, index it in Chroma, chat with the document, then evaluate using an Excel file with questions and reference answers.</p>

      <section>
        <h2>1. Ingest PDF</h2>
        <input type="file" accept=".pdf" onChange={e => setPdf(e.target.files[0])} />
        <button onClick={ingestPdf}>Upload & Index PDF</button>
      </section>

      <section className="chat-section">
        <div className="chat-header">
          <div>
            <h2>2. Chat with the PDF</h2>
            <span className="session">Session {sessionId.slice(0, 8)}</span>
          </div>
          <label>Top K <input type="number" value={topK} min="1" max="10" onChange={e => setTopK(e.target.value)} /></label>
        </div>

        <div className="conversation">
          {messages.length === 0 && (
            <div className="empty-chat">Ask a question from the indexed PDF. Follow-up questions will use the prior conversation.</div>
          )}
          {messages.map((message, index) => (
            <article key={index} className={`message ${message.role}`}>
              <div className="bubble">
                <div className="role">{message.role === 'user' ? 'You' : 'Assistant'}</div>
                <p>{message.content}</p>
                {message.piiRedactionLog?.length > 0 && (
                  <div className="meta">{message.piiRedactionLog.join('; ')}</div>
                )}
                {message.answerPiiRedactionLog?.length > 0 && (
                  <div className="meta">{message.answerPiiRedactionLog.join('; ')}</div>
                )}
                {message.contexts?.length > 0 && (
                  <details>
                    <summary>Retrieved chunks</summary>
                    {message.contexts.map((context, chunkIndex) => <pre key={chunkIndex}>{context}</pre>)}
                  </details>
                )}
              </div>
            </article>
          ))}
          {asking && <div className="message assistant"><div className="bubble"><div className="role">Assistant</div><p>Thinking...</p></div></div>}
        </div>

        <div className="composer">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={onQuestionKeyDown}
            placeholder="Ask a question or a follow-up..."
          />
          <div className="composer-actions">
            <button onClick={ask} disabled={asking}>{asking ? 'Asking...' : 'Send'}</button>
            <button className="secondary" onClick={clearConversation} disabled={asking || messages.length === 0}>New Chat</button>
          </div>
        </div>
      </section>

      <section>
        <h2>3. Run RAGAS Evaluation</h2>
        <p>Excel columns: <b>question</b> and one of <b>reference_answer</b>, <b>reference</b>, <b>ground_truth</b>, or <b>answer</b>.</p>
        <input type="file" accept=".xlsx,.xls" onChange={e => setExcel(e.target.files[0])} />
        <button onClick={evaluate}>Run Evaluation</button>
        {download && <a className="download" href={download}>Download Evaluation Excel</a>}
      </section>

      <section>
        <h2>Logs</h2>
        <pre>{log}</pre>
      </section>
    </div>
  );
}
