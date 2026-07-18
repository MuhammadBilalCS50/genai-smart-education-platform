import axios from 'axios';
import { useEffect, useMemo, useState } from 'react';

const API = 'http://localhost:8000';

function makeSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function Landing({ onOpen }) {
  return (
    <main className="landing-page">
      <div className="landing-content">
        <span className="landing-eyebrow">Smart Education Platform</span>
        <h1>Choose your learning assistant</h1>
        <p>Select a workspace to get started.</p>
        <div className="assistant-grid">
          <button className="assistant-card" type="button" onClick={() => onOpen('student-rag')}>
            <span className="assistant-card-icon" aria-hidden="true">S</span>
            <span className="assistant-card-copy">
              <strong>Student RAG Assistant</strong>
              <span>Upload study material and ask grounded questions about it.</span>
            </span>
            <span className="assistant-card-arrow" aria-hidden="true">→</span>
          </button>
          <button className="assistant-card quiz-card" type="button" onClick={() => onOpen('quiz')}>
            <span className="assistant-card-icon" aria-hidden="true">Q</span>
            <span className="assistant-card-copy">
              <strong>AI Quiz Generator</strong>
              <span>Create section-based short-question papers with answers and sources.</span>
            </span>
            <span className="assistant-card-arrow" aria-hidden="true">→</span>
          </button>
        </div>
      </div>
    </main>
  );
}

function PageHeader({ title, description, onBack }) {
  return (
    <header className="assistant-page-header">
      <div><h1>{title}</h1><p>{description}</p></div>
      <button className="secondary back-button" type="button" onClick={onBack}>Back to home</button>
    </header>
  );
}

function StudentAssistant({ onBack }) {
  const [pdf, setPdf] = useState(null);
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(4);
  const [log, setLog] = useState('');
  const [messages, setMessages] = useState([]);
  const [asking, setAsking] = useState(false);
  const sessionId = useMemo(makeSessionId, []);
  const append = (text) => setLog(prev => `${prev}\n${text}`.trim());

  async function ingestPdf() {
    if (!pdf) return alert('Select a PDF first');
    const form = new FormData();
    form.append('file', pdf);
    append('Indexing PDF...');
    try {
      const res = await axios.post(`${API}/ingest-pdf`, form);
      append(JSON.stringify(res.data, null, 2));
    } catch (err) {
      append(`Upload failed: ${err.response?.data?.detail || err.message}`);
    }
  }

  async function ask() {
    const text = question.trim();
    if (!text) return alert('Enter a question');
    setQuestion('');
    setAsking(true);
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    try {
      const res = await axios.post(`${API}/ask`, { question: text, top_k: Number(topK), session_id: sessionId });
      setMessages(prev => [...prev, {
        role: 'assistant', content: res.data.answer, queryType: res.data.query_type,
        references: res.data.references || [], contexts: res.data.contexts || [],
        piiRedactionLog: res.data.pii_redaction_log || [],
        answerPiiRedactionLog: res.data.answer_pii_redaction_log || [],
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.response?.data?.detail || err.message}` }]);
    } finally { setAsking(false); }
  }

  async function clearConversation() {
    setMessages([]); setQuestion('');
    await axios.delete(`${API}/chat/${sessionId}`).catch(() => null);
  }

  return (
    <div className="container">
      <PageHeader title="Student RAG Assistant" description="Upload a PDF, index it in Chroma, and chat with the document." onBack={onBack} />
      <section><h2>1. Ingest PDF</h2><input type="file" accept=".pdf" onChange={e => setPdf(e.target.files[0])} /><button onClick={ingestPdf}>Upload & Index PDF</button></section>
      <section className="chat-section">
        <div className="chat-header"><div><h2>2. Chat with the PDF</h2><span className="session">Session {sessionId.slice(0, 8)}</span></div><label>Top K <input type="number" value={topK} min="1" max="10" onChange={e => setTopK(e.target.value)} /></label></div>
        <div className="conversation">
          {messages.length === 0 && <div className="empty-chat">Ask a question from the indexed PDF. Follow-up questions will use the prior conversation.</div>}
          {messages.map((message, index) => (
            <article key={index} className={`message ${message.role}`}><div className="bubble">
              <div className="role">{message.role === 'user' ? 'You' : 'Assistant'}</div>
              {message.queryType && <div className="query-type">Route: {message.queryType}</div>}
              <p>{message.content}</p>
              {message.piiRedactionLog?.length > 0 && <div className="meta">{message.piiRedactionLog.join('; ')}</div>}
              {message.answerPiiRedactionLog?.length > 0 && <div className="meta">{message.answerPiiRedactionLog.join('; ')}</div>}
              {message.references?.length > 0 && <div className="references"><strong>Relevant concepts in the document</strong><ul>{message.references.map((ref, i) => <li key={`${ref.heading}-${ref.pages}-${i}`}>{ref.heading} — {ref.pages === 'Page unavailable' ? ref.pages : `Page(s): ${ref.pages}`}</li>)}</ul></div>}
              {message.contexts?.length > 0 && <details><summary>Retrieved chunks</summary>{message.contexts.map((context, i) => <pre key={i}>{context}</pre>)}</details>}
            </div></article>
          ))}
          {asking && <div className="message assistant"><div className="bubble"><div className="role">Assistant</div><p>Thinking...</p></div></div>}
        </div>
        <div className="composer"><textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!asking) ask(); } }} placeholder="Ask a question or a follow-up..." /><div className="composer-actions"><button onClick={ask} disabled={asking}>{asking ? 'Asking...' : 'Send'}</button><button className="secondary" onClick={clearConversation} disabled={asking || messages.length === 0}>New Chat</button></div></div>
      </section>
      <section><h2>Logs</h2><pre>{log}</pre></section>
    </div>
  );
}

function QuizGenerator({ onBack }) {
  const [books, setBooks] = useState([]);
  const [bookId, setBookId] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [actualFirstPage, setActualFirstPage] = useState('');
  const [selectedSections, setSelectedSections] = useState([]);
  const [difficulty, setDifficulty] = useState('medium');
  const [instructions, setInstructions] = useState('');
  const [quiz, setQuiz] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    setBusy('Loading indexed books...');
    axios.get(`${API}/quiz/books`)
      .then(res => setBooks(res.data.books || []))
      .catch(err => setError(err.response?.data?.detail || err.message))
      .finally(() => setBusy(''));
  }, []);

  function resetAfterBook(nextBookId) {
    setBookId(nextBookId); setAnalysis(null); setCalibration(null); setQuiz(null);
    setSelectedSections([]); setActualFirstPage(''); setError('');
  }

  async function analyzeBook() {
    if (!bookId) return setError('Select a book first.');
    setBusy('Reading the table of contents with AI...'); setError(''); setQuiz(null);
    try {
      const { data } = await axios.post(`${API}/quiz/contents`, { book_id: bookId });
      setAnalysis(data); setActualFirstPage(String(data.sections[0].start_page));
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  async function calibratePages() {
    if (!actualFirstPage || Number(actualFirstPage) < 1) return setError('Enter the actual PDF page of the first section.');
    setBusy('Saving the page correction...'); setError('');
    try {
      const { data } = await axios.post(`${API}/quiz/calibrate`, { analysis_id: analysis.analysis_id, actual_first_page: Number(actualFirstPage) });
      setCalibration(data); setSelectedSections([]); setQuiz(null);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  function toggleSection(id) {
    setSelectedSections(prev => prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]);
  }

  async function generateQuiz() {
    if (selectedSections.length === 0) return setError('Select at least one section.');
    setBusy('Generating and answering the quiz. This can take a few minutes...'); setError(''); setQuiz(null);
    try {
      const { data } = await axios.post(`${API}/quiz/generate`, {
        calibration_id: calibration.calibration_id,
        selected_section_ids: selectedSections,
        difficulty,
        instructions,
      });
      setQuiz(data);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  return (
    <div className="container quiz-page">
      <PageHeader title="AI Quiz Generator" description="Build a grounded short-question quiz from selected book sections." onBack={onBack} />
      {error && <div className="alert error" role="alert">{error}</div>}
      {busy && <div className="alert progress" role="status"><span className="spinner" />{busy}</div>}

      <section>
        <div className="step-heading"><span>1</span><div><h2>Choose a book</h2><p>Books are loaded from Markdown files already created during ingestion.</p></div></div>
        {books.length === 0 && !busy ? <p className="muted">No indexed Markdown books were found.</p> : <div className="inline-form"><select value={bookId} onChange={e => resetAfterBook(e.target.value)}><option value="">Select an indexed book</option>{books.map(book => <option key={book.id} value={book.id}>{book.name}</option>)}</select><button onClick={analyzeBook} disabled={!bookId || Boolean(busy)}>Analyze contents</button></div>}
      </section>

      {analysis && <section>
        <div className="step-heading"><span>2</span><div><h2>Confirm page numbering</h2><p>The contents uses printed page numbers. Tell us where its first section actually begins in the PDF.</p></div></div>
        <div className="table-wrap"><table><thead><tr><th>Section</th><th>Contents start</th><th>Contents end</th></tr></thead><tbody>{analysis.sections.map(section => <tr key={section.id}><td>{section.title}</td><td>{section.start_page}</td><td>{section.end_page}</td></tr>)}</tbody></table></div>
        <label className="field compact">Actual PDF page for “{analysis.sections[0].title}”<input type="number" min="1" value={actualFirstPage} onChange={e => { setActualFirstPage(e.target.value); setCalibration(null); }} /></label>
        <button onClick={calibratePages} disabled={Boolean(busy)}>Confirm page correction</button>
      </section>}

      {calibration && <section>
        <div className="step-heading"><span>3</span><div><h2>Configure your quiz</h2><p>Page correction saved: {calibration.delta >= 0 ? '+' : ''}{calibration.delta} page{Math.abs(calibration.delta) === 1 ? '' : 's'}.</p></div></div>
        <div className="section-picker">{calibration.sections.map(section => <label key={section.id} className={`section-option ${selectedSections.includes(section.id) ? 'selected' : ''}`}><input type="checkbox" checked={selectedSections.includes(section.id)} onChange={() => toggleSection(section.id)} /><span><strong>{section.title}</strong><small>PDF pages {section.actual_start_page}–{section.actual_end_page}</small></span></label>)}</div>
        <div className="quiz-settings"><label className="field">Difficulty<select value={difficulty} onChange={e => setDifficulty(e.target.value)}><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select></label><label className="field">Additional instructions (optional)<textarea value={instructions} onChange={e => setInstructions(e.target.value)} placeholder="For example: Generate 15 questions and focus on definitions." /></label></div>
        <button onClick={generateQuiz} disabled={Boolean(busy) || selectedSections.length === 0}>Generate quiz</button>
      </section>}

      {quiz && <section className="quiz-result">
        <div className="step-heading success"><span>✓</span><div><h2>Your quiz is ready</h2><p>{quiz.questions.length} short questions generated with grounded answers and source metadata.</p></div></div>
        <div className="download-actions"><a className="button-link" href={`${API}${quiz.downloads.questions}`}>Download question paper</a><a className="button-link secondary-link" href={`${API}${quiz.downloads.answers}`}>Download paper with answers</a></div>
        <details><summary>Preview questions and answers</summary><ol className="quiz-preview">{quiz.questions.map((item, index) => <li key={index}><strong>{item.question}</strong><p>{item.answer}</p>{item.references?.length > 0 && <small>{item.references.map(ref => `${ref.heading} — page(s) ${ref.pages}`).join('; ')}</small>}</li>)}</ol></details>
      </section>}
    </div>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState('landing');
  if (activeView === 'landing') return <Landing onOpen={setActiveView} />;
  if (activeView === 'quiz') return <QuizGenerator onBack={() => setActiveView('landing')} />;
  return <StudentAssistant onBack={() => setActiveView('landing')} />;
}
