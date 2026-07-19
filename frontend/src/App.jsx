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
          <button className="assistant-card slides-card" type="button" onClick={() => onOpen('slides')}>
            <span className="assistant-card-icon" aria-hidden="true">P</span>
            <span className="assistant-card-copy">
              <strong>AI Slides Generator</strong>
              <span>Turn selected book sections into a reviewed, downloadable presentation.</span>
            </span>
            <span className="assistant-card-arrow" aria-hidden="true">→</span>
          </button>
          <button className="assistant-card checker-card" type="button" onClick={() => onOpen('paper-checker')}>
            <span className="assistant-card-icon" aria-hidden="true">C</span>
            <span className="assistant-card-copy">
              <strong>AI Paper Checker</strong>
              <span>Read handwritten answers, apply a mark scheme, review marks, and export a report.</span>
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

function SlidesGenerator({ onBack }) {
  const [books, setBooks] = useState([]);
  const [bookId, setBookId] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [actualFirstPage, setActualFirstPage] = useState('');
  const [selectedSections, setSelectedSections] = useState([]);
  const [slideCount, setSlideCount] = useState(10);
  const [audience, setAudience] = useState('Students');
  const [instructions, setInstructions] = useState('');
  const [generateImages, setGenerateImages] = useState(false);
  const [draft, setDraft] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [presentation, setPresentation] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    setBusy('Loading indexed books...');
    axios.get(`${API}/slides/books`)
      .then(res => setBooks(res.data.books || []))
      .catch(err => setError(err.response?.data?.detail || err.message))
      .finally(() => setBusy(''));
  }, []);

  function resetAfterBook(nextBookId) {
    setBookId(nextBookId); setAnalysis(null); setCalibration(null); setDraft(null);
    setPresentation(null); setSelectedSections([]); setActualFirstPage(''); setError('');
  }

  async function analyzeBook() {
    if (!bookId) return setError('Select a book first.');
    setBusy('Reading the table of contents with AI...'); setError('');
    try {
      const { data } = await axios.post(`${API}/slides/contents`, { book_id: bookId });
      setAnalysis(data); setActualFirstPage(String(data.sections[0].start_page));
      setCalibration(null); setDraft(null); setPresentation(null);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  async function calibratePages() {
    if (!actualFirstPage || Number(actualFirstPage) < 1) return setError('Enter the actual PDF page of the first section.');
    setBusy('Saving the page correction...'); setError('');
    try {
      const { data } = await axios.post(`${API}/slides/calibrate`, {
        analysis_id: analysis.analysis_id, actual_first_page: Number(actualFirstPage),
      });
      setCalibration(data); setSelectedSections([]); setDraft(null); setPresentation(null);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  function toggleSection(id) {
    setSelectedSections(prev => prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]);
  }

  async function generateDraft() {
    if (selectedSections.length === 0) return setError('Select at least one section.');
    setBusy(generateImages
      ? 'Designing slides and generating optional images. This can take several minutes...'
      : 'Retrieving book content and designing your slide draft...');
    setError('');
    try {
      const { data } = await axios.post(`${API}/slides/generate`, {
        calibration_id: calibration.calibration_id,
        selected_section_ids: selectedSections,
        slide_count: Number(slideCount), audience, instructions, generate_images: generateImages,
      });
      setDraft(data); setFeedback(''); setPresentation(null);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  async function reviseDraft() {
    if (!feedback.trim()) return setError('Describe what you want changed.');
    setBusy(draft.generate_images
      ? 'Revising slides and updating changed images...'
      : 'Incorporating your feedback...');
    setError('');
    try {
      const { data } = await axios.post(`${API}/slides/${draft.draft_id}/feedback`, { feedback });
      setDraft(data); setFeedback(''); setPresentation(null);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  async function exportPresentation() {
    setBusy('Rendering the final PowerPoint...'); setError('');
    try {
      const { data } = await axios.post(`${API}/slides/${draft.draft_id}/export`);
      setPresentation(data);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  return (
    <div className="container slides-page">
      <PageHeader title="AI Slides Generator" description="Create, review, revise, and export grounded slides from your books." onBack={onBack} />
      {error && <div className="alert error" role="alert">{error}</div>}
      {busy && <div className="alert progress" role="status"><span className="spinner" />{busy}</div>}

      <section>
        <div className="step-heading slides-step"><span>1</span><div><h2>Choose a book</h2><p>Select an indexed book and extract its teaching sections.</p></div></div>
        {books.length === 0 && !busy ? <p className="muted">No indexed Markdown books were found.</p> : <div className="inline-form"><select value={bookId} onChange={e => resetAfterBook(e.target.value)}><option value="">Select an indexed book</option>{books.map(book => <option key={book.id} value={book.id}>{book.name}</option>)}</select><button onClick={analyzeBook} disabled={!bookId || Boolean(busy)}>Analyze contents</button></div>}
      </section>

      {analysis && <section>
        <div className="step-heading slides-step"><span>2</span><div><h2>Confirm page numbering</h2><p>Match the first printed section page to its actual PDF page.</p></div></div>
        <div className="table-wrap"><table><thead><tr><th>Section</th><th>Printed start</th><th>Printed end</th></tr></thead><tbody>{analysis.sections.map(section => <tr key={section.id}><td>{section.title}</td><td>{section.start_page}</td><td>{section.end_page}</td></tr>)}</tbody></table></div>
        <label className="field compact">Actual PDF page for “{analysis.sections[0].title}”<input type="number" min="1" value={actualFirstPage} onChange={e => { setActualFirstPage(e.target.value); setCalibration(null); setDraft(null); }} /></label>
        <button onClick={calibratePages} disabled={Boolean(busy)}>Confirm page correction</button>
      </section>}

      {calibration && <section>
        <div className="step-heading slides-step"><span>3</span><div><h2>Select content and configure the deck</h2><p>Page correction: {calibration.delta >= 0 ? '+' : ''}{calibration.delta}. Choose the material the presentation should teach.</p></div></div>
        <div className="section-picker">{calibration.sections.map(section => <label key={section.id} className={`section-option slide-option ${selectedSections.includes(section.id) ? 'selected' : ''}`}><input type="checkbox" checked={selectedSections.includes(section.id)} onChange={() => toggleSection(section.id)} /><span><strong>{section.title}</strong><small>PDF pages {section.actual_start_page}–{section.actual_end_page}</small></span></label>)}</div>
        <div className="slides-settings">
          <label className="field">Slides<input type="number" min="3" max="30" value={slideCount} onChange={e => setSlideCount(e.target.value)} /></label>
          <label className="field">Audience<input type="text" value={audience} onChange={e => setAudience(e.target.value)} placeholder="For example: Grade 10 students" /></label>
          <label className="field wide">Learning goal and instructions<textarea value={instructions} onChange={e => setInstructions(e.target.value)} placeholder="For example: Explain the core concepts with examples and finish with a recap." /></label>
          <label className="image-generation-option wide">
            <input type="checkbox" checked={generateImages} onChange={e => { setGenerateImages(e.target.checked); setDraft(null); setPresentation(null); }} />
            <span><strong>Generate image-based slides with GPT Image 2</strong><small>Optional and billed separately. Every slide will contain only a generated 3:2 image with no additional text. Uses the low-quality setting to minimize cost.</small></span>
          </label>
        </div>
        <button onClick={generateDraft} disabled={Boolean(busy) || selectedSections.length === 0}>Generate slide draft</button>
      </section>}

      {draft && <section className="slides-result">
        <div className="step-heading slides-step"><span>4</span><div><h2>Review the draft</h2><p>{draft.slides.length} slides · Revision {draft.revision}{draft.generate_images ? ` · ${draft.images_generated}/${draft.slides.length} images generated` : ''} · Theme: {draft.theme_recommendation}</p></div></div>
        {draft.image_generation_failures > 0 && <div className="alert image-warning">{draft.image_generation_failures} image{draft.image_generation_failures === 1 ? '' : 's'} could not be generated. Those slides will remain blank in the image-only export.</div>}
        <div className="slide-preview-grid">{draft.slides.map((slide, index) => <article className="slide-preview" key={`${draft.revision}-${index}`}>
          <div className="slide-preview-top"><span>{index + 1}</span><small>{slide.layout_recommendation.replace('_', ' ')}</small></div>
          <h3>{slide.title}</h3>{slide.subtitle && <p className="slide-subtitle">{slide.subtitle}</p>}
          {slide.bullets?.length > 0 && <ul>{slide.bullets.map((bullet, itemIndex) => <li key={itemIndex}>{bullet}</li>)}</ul>}
          {slide.picture_recommendation && <div className="visual-recommendation"><strong>Picture / diagram {slide.image_generated ? '· Image generated' : ''}</strong><span>{slide.picture_recommendation}</span>{slide.image_generation_error && <em>{slide.image_generation_error}</em>}</div>}
          {slide.source_pages && <small className="source-pages">Source PDF page(s): {slide.source_pages}</small>}
        </article>)}</div>
        <div className="feedback-panel"><label className="field">Your feedback<textarea value={feedback} onChange={e => setFeedback(e.target.value)} placeholder="For example: Make slide 4 simpler, add a comparison slide, and use fewer bullets." /></label><button onClick={reviseDraft} disabled={Boolean(busy) || !feedback.trim()}>Revise draft</button><button className="export-button" onClick={exportPresentation} disabled={Boolean(busy)}>Approve and export PowerPoint</button></div>
      </section>}

      {presentation && <section>
        <div className="step-heading success"><span>✓</span><div><h2>Your presentation is ready</h2><p>{draft?.generate_images ? 'The approved draft was rendered as a 3:2 image-only PowerPoint file.' : 'The approved draft was rendered as a widescreen PowerPoint file.'}</p></div></div>
        <a className="button-link slides-download" href={`${API}${presentation.download}`}>Download {presentation.filename}</a>
      </section>}
    </div>
  );
}

function PaperChecker({ onBack }) {
  const [paperFile, setPaperFile] = useState(null);
  const [schemeFile, setSchemeFile] = useState(null);
  const [paper, setPaper] = useState(null);
  const [scheme, setScheme] = useState(null);
  const [check, setCheck] = useState(null);
  const [reviewedMarks, setReviewedMarks] = useState({});
  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  async function parseUpload(kind) {
    const file = kind === 'paper' ? paperFile : schemeFile;
    if (!file) return setError(`Select a ${kind === 'paper' ? 'student paper' : 'mark scheme'} PDF first.`);
    setBusy(kind === 'paper'
      ? 'Running PaddleOCR on the handwritten paper and identifying questions and answers...'
      : 'Reading the mark scheme and extracting its rubrics with AI...');
    setError(''); setCheck(null); setReport(null);
    const form = new FormData(); form.append('file', file);
    try {
      const endpoint = kind === 'paper' ? 'paper' : 'mark-scheme';
      const { data } = await axios.post(`${API}/paper-checker/${endpoint}`, form);
      if (kind === 'paper') setPaper(data); else setScheme(data);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  async function checkPaper() {
    if (!paper || !scheme) return setError('Parse both PDFs before checking the paper.');
    setBusy('Classifying questions and marking short questions against the rubric...');
    setError(''); setReport(null);
    try {
      const { data } = await axios.post(`${API}/paper-checker/check`, {
        paper_id: paper.paper_id, mark_scheme_id: scheme.mark_scheme_id,
      });
      setCheck(data);
      setReviewedMarks(Object.fromEntries(data.assessments.map(item => [item.question_number, item.awarded_marks])));
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  function changeMark(item, value) {
    setReviewedMarks(previous => ({
      ...previous, [item.question_number]: value === '' ? '' : Number(value),
    }));
  }

  async function submitMarks() {
    const invalid = check.assessments.find(item => {
      const value = reviewedMarks[item.question_number];
      return value === '' || !Number.isFinite(Number(value)) || Number(value) < 0 || Number(value) > item.max_marks;
    });
    if (invalid) return setError(`Enter marks from 0 to ${invalid.max_marks} for question ${invalid.question_number}.`);
    setBusy('Finalizing totals and creating the marks report...'); setError('');
    try {
      const { data } = await axios.post(`${API}/paper-checker/${check.check_id}/submit`, {
        marks: check.assessments.map(item => ({
          question_number: item.question_number,
          awarded_marks: Number(reviewedMarks[item.question_number]),
        })),
      });
      setReport(data);
    } catch (err) { setError(err.response?.data?.detail || err.message); }
    finally { setBusy(''); }
  }

  return (
    <div className="container checker-page">
      <PageHeader title="AI Paper Checker" description="OCR a solved paper, apply its mark scheme, review AI marks, and export the final report." onBack={onBack} />
      {error && <div className="alert error" role="alert">{error}</div>}
      {busy && <div className="alert progress" role="status"><span className="spinner" />{busy}</div>}

      <section>
        <div className="step-heading checker-step"><span>1</span><div><h2>Upload the solved paper</h2><p>Use a scanned PDF with typed questions and clear handwritten answers.</p></div></div>
        <div className="upload-row"><input type="file" accept="application/pdf,.pdf" onChange={e => { setPaperFile(e.target.files[0] || null); setPaper(null); setCheck(null); setReport(null); }} /><button onClick={() => parseUpload('paper')} disabled={!paperFile || Boolean(busy)}>Extract paper</button></div>
        {paper && <div className="parse-success"><strong>{paper.questions.length} questions extracted</strong><span>{paper.source_filename}</span></div>}
      </section>

      <section>
        <div className="step-heading checker-step"><span>2</span><div><h2>Upload the mark scheme</h2><p>The scheme may contain expected points, rubrics, examiner notes, and per-question marks.</p></div></div>
        <div className="upload-row"><input type="file" accept="application/pdf,.pdf" onChange={e => { setSchemeFile(e.target.files[0] || null); setScheme(null); setCheck(null); setReport(null); }} /><button onClick={() => parseUpload('scheme')} disabled={!schemeFile || Boolean(busy)}>Extract mark scheme</button></div>
        {scheme && <div className="parse-success"><strong>{scheme.items.length} marking entries extracted</strong><span>{scheme.source_filename}</span></div>}
      </section>

      {paper && scheme && <section>
        <div className="step-heading checker-step"><span>3</span><div><h2>Check short questions</h2><p>Other detected question types are classified internally and excluded from marking.</p></div></div>
        <button className="checker-button" onClick={checkPaper} disabled={Boolean(busy)}>Classify and check paper</button>
      </section>}

      {check && !report && <section>
        <div className="step-heading checker-step"><span>4</span><div><h2>Review the proposed marks</h2><p>Change any mark if needed, then submit once to calculate the final total.</p></div></div>
        <div className="assessment-list">{check.assessments.map(item => <article className="assessment-card" key={item.question_number}>
          <div className="assessment-heading"><strong>Question {item.question_number}</strong><label>Marks <input type="number" min="0" max={item.max_marks} step="0.5" value={reviewedMarks[item.question_number] ?? ''} onChange={e => changeMark(item, e.target.value)} /> / {item.max_marks}</label></div>
          <p className="question-copy">{item.question_text}</p>
          <div className="answer-copy"><strong>Student answer</strong><p>{item.answer_text || 'No answer recognized.'}</p></div>
          <div className="mark-reason"><strong>AI marking reason</strong><p>{item.reason}</p></div>
          <details><summary>View applied mark scheme</summary><p>{item.mark_scheme}</p></details>
        </article>)}</div>
        <button className="checker-button" onClick={submitMarks} disabled={Boolean(busy)}>Submit reviewed marks</button>
      </section>}

      {report && <section>
        <div className="step-heading success"><span>✓</span><div><h2>Marks report ready</h2><p>Final score: {report.total_awarded} / {report.total_possible} ({report.percentage.toFixed(1)}%)</p></div></div>
        <a className="button-link checker-download" href={`${API}${report.report}`}>Download PDF marks report</a>
      </section>}
    </div>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState('landing');
  if (activeView === 'landing') return <Landing onOpen={setActiveView} />;
  if (activeView === 'quiz') return <QuizGenerator onBack={() => setActiveView('landing')} />;
  if (activeView === 'slides') return <SlidesGenerator onBack={() => setActiveView('landing')} />;
  if (activeView === 'paper-checker') return <PaperChecker onBack={() => setActiveView('landing')} />;
  return <StudentAssistant onBack={() => setActiveView('landing')} />;
}
