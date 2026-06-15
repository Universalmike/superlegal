import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const DOCUMENT_TYPES = [
  { icon: '📜', label: 'Demand Letter' },
  { icon: '🏠', label: 'Quit Notice' },
  { icon: '⚖️', label: 'Formal Complaint Letter' },
  { icon: '📋', label: 'Employment Rights Notice' },
  { icon: '📝', label: 'Statement of Facts' },
];

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('All Documents');
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState('Chat 1');
  const [messages, setMessages] = useState([]);
  const [language, setLanguage] = useState('English');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('GEMINI_API_KEY') || '');
  const [newSessionName, setNewSessionName] = useState('');
  const [references, setReferences] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');

  // PDF viewer state
  const [isPdfOpen, setIsPdfOpen] = useState(false);
  const [activePdfUrl, setActivePdfUrl] = useState('');
  const [activePdfPage, setActivePdfPage] = useState(1);
  const [activePdfDocName, setActivePdfDocName] = useState('');
  const [activeRefIdx, setActiveRefIdx] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  // Draft document state
  const [draftPopoverIdx, setDraftPopoverIdx] = useState(null); // which AI message has popover open
  const [draftLoading, setDraftLoading] = useState(false);
  const [draftModal, setDraftModal] = useState(null); // { document, document_type, generated_at }
  const [copySuccess, setCopySuccess] = useState(false);

  const messagesEndRef = useRef(null);
  const popoverRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('GEMINI_API_KEY', apiKey);
  }, [apiKey]);

  useEffect(() => {
    fetchDocuments();
    fetchSessions();
  }, []);

  useEffect(() => {
    if (activeSession) {
      fetchSessionHistory(activeSession);
      setReferences([]);
      setIsPdfOpen(false);
      setActiveRefIdx(null);
      setDraftPopoverIdx(null);
    }
  }, [activeSession]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Close popover when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setDraftPopoverIdx(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
        setSelectedDoc('All Documents');
      }
    } catch (err) {
      console.error('Failed to fetch documents', err);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions`);
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        if (data.length > 0 && !data.includes(activeSession)) {
          setActiveSession(data[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch sessions', err);
    }
  };

  const fetchSessionHistory = async (sessionId) => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data);
      }
    } catch (err) {
      console.error('Failed to fetch session history', err);
    }
  };

  const handleCreateSession = async (e) => {
    e.preventDefault();
    if (!newSessionName.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: newSessionName.trim() }),
      });
      if (res.ok) {
        setNewSessionName('');
        await fetchSessions();
        setActiveSession(newSessionName.trim());
        setErrorMsg('');
      } else {
        const errData = await res.json();
        setErrorMsg(errData.detail || 'Failed to create session.');
      }
    } catch {
      setErrorMsg('Network error creating session.');
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;
    if (!apiKey) {
      setErrorMsg('Please enter your Google Gemini API Key in the sidebar settings.');
      return;
    }

    setErrorMsg('');
    const userQuery = query.trim();
    setQuery('');
    setMessages((prev) => [...prev, { sender: 'user', text: userQuery }]);
    setLoading(true);
    setReferences([]);
    setIsPdfOpen(false);
    setActiveRefIdx(null);
    setDraftPopoverIdx(null);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession,
          document_key: selectedDoc,
          language,
          query: userQuery,
          api_key: apiKey,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setMessages((prev) => [...prev, { sender: 'ai', text: data.answer }]);
        setReferences(data.references || []);
      } else {
        const errData = await res.json();
        setErrorMsg(errData.detail || 'An error occurred.');
        setMessages((prev) => prev.slice(0, -1));
      }
    } catch {
      setErrorMsg('Network error communicating with backend.');
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const handleOpenReference = (ref, idx) => {
    const pdfUrl = `${API_BASE}/documents/${encodeURIComponent(ref.filename)}#page=${ref.page}`;
    setActivePdfUrl(pdfUrl);
    setActivePdfPage(ref.page);
    setActivePdfDocName(ref.document_name);
    setActiveRefIdx(idx);
    setPdfLoading(true);
    setIsPdfOpen(true);
    setDraftPopoverIdx(null);
  };

  const handleDraftDocument = async (documentType) => {
    setDraftPopoverIdx(null);
    setDraftLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/draft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession,
          document_type: documentType,
          api_key: apiKey,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setDraftModal(data);
      } else {
        const errData = await res.json();
        setErrorMsg(errData.detail || 'Document generation failed.');
      }
    } catch {
      setErrorMsg('Network error during document generation.');
    } finally {
      setDraftLoading(false);
    }
  };

  const handleCopyDocument = async () => {
    if (!draftModal) return;
    try {
      await navigator.clipboard.writeText(draftModal.document);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2500);
    } catch {
      setErrorMsg('Failed to copy to clipboard.');
    }
  };

  const handleDownloadDocument = () => {
    if (!draftModal) return;
    const filename = `${draftModal.document_type.replace(/\s+/g, '_')}_${new Date().toISOString().slice(0, 10)}.txt`;
    const blob = new Blob([draftModal.document], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Count AI messages for popover index tracking
  const getAiMessageIdx = (messages, currentIdx) => {
    return messages.slice(0, currentIdx + 1).filter((m) => m.sender === 'ai').length - 1;
  };

  return (
    <div className="app-container">
      {/* ── SIDEBAR ── */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">⚖️ Super Legal</h1>
          <div className="settings-group">
            {errorMsg && <div className="alert-warning">{errorMsg}</div>}
            <label className="settings-label">Gemini API Key</label>
            <input
              id="api-key-input"
              type="password"
              className="text-input"
              placeholder="Paste your API key here..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <label className="settings-label">Search Scope</label>
            <select
              id="document-selector"
              className="select-input"
              value={selectedDoc}
              onChange={(e) => setSelectedDoc(e.target.value)}
            >
              {documents.map((doc) => (
                <option key={doc} value={doc}>
                  {doc === 'All Documents' ? '🔍 All Documents (Auto-Detect)' : doc}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="sessions-list-container">
          <div className="sessions-title">Chat Sessions</div>
          {sessions.map((sess) => (
            <div
              key={sess}
              id={`session-${sess.replace(/\s+/g, '-').toLowerCase()}`}
              className={`session-item ${activeSession === sess ? 'active' : ''}`}
              onClick={() => setActiveSession(sess)}
            >
              <span className="session-icon">💬</span>
              <span className="session-name">{sess}</span>
            </div>
          ))}
        </div>

        <form className="new-session-form" onSubmit={handleCreateSession}>
          <input
            id="new-session-input"
            type="text"
            className="text-input"
            placeholder="New chat name..."
            value={newSessionName}
            onChange={(e) => setNewSessionName(e.target.value)}
            style={{ marginBottom: '8px' }}
          />
          <button id="create-session-btn" type="submit" className="btn-primary" style={{ width: '100%' }}>
            + Create New Chat
          </button>
        </form>
      </div>

      {/* ── CHAT AREA ── */}
      <div className="chat-area">
        <div className="chat-header">
          <div className="chat-header-info">
            <h2 className="chat-header-title">{activeSession}</h2>
            <div className="chat-header-subtitle">
              {selectedDoc === 'All Documents'
                ? <span>🔍 Searching across <span style={{ color: '#00ffaa' }}>all Nigerian laws</span></span>
                : <span>Querying: <span style={{ color: '#00ffaa' }}>{selectedDoc}</span></span>
              }
            </div>
          </div>
          <div className="chat-header-controls">
            <label className="settings-label" style={{ margin: 0 }}>Language:</label>
            <select
              id="language-selector"
              className="select-input"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              style={{ width: '110px' }}
            >
              <option value="English">English</option>
              <option value="Pidgin">Pidgin</option>
              <option value="Yoruba">Yoruba</option>
              <option value="Hausa">Hausa</option>
              <option value="Igbo">Igbo</option>
              <option value="French">French</option>
            </select>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 && !loading && (
            <div style={{ margin: 'auto', textAlign: 'center', opacity: 0.6, maxWidth: '420px' }}>
              <span style={{ fontSize: '3rem' }}>⚖️</span>
              <h3>Welcome to Super Legal</h3>
              <p style={{ fontSize: '0.9rem', lineHeight: 1.6 }}>
                Ask any question about Nigerian law. D Law searches across the <strong>Constitution</strong>, <strong>Labour Act</strong>, and <strong>Criminal Code</strong> and cites its sources. You can also <strong>generate legal documents</strong> from your conversation.
              </p>
            </div>
          )}

          {messages.map((msg, index) => {
            const aiIdx = msg.sender === 'ai' ? getAiMessageIdx(messages, index) : null;
            return (
              <div key={index} className={`message-bubble ${msg.sender}`}>
                <div className="message-bubble-header">
                  {msg.sender === 'user' ? '🧑‍⚖️ Client' : '🤖 D Law'}
                </div>
                <div className="message-text">{msg.text}</div>

                {/* Draft action row — only shown on AI messages */}
                {msg.sender === 'ai' && (
                  <div className="message-actions">
                    <button
                      id={`draft-btn-${aiIdx}`}
                      className="btn-draft"
                      disabled={draftLoading}
                      onClick={() => setDraftPopoverIdx(draftPopoverIdx === aiIdx ? null : aiIdx)}
                    >
                      {draftLoading ? '⏳ Drafting...' : '📝 Draft Document'}
                    </button>

                    {/* Document type popover */}
                    {draftPopoverIdx === aiIdx && (
                      <div className="draft-popover" ref={popoverRef}>
                        <div className="draft-popover-title">Choose document type</div>
                        {DOCUMENT_TYPES.map((dt) => (
                          <button
                            key={dt.label}
                            id={`draft-type-${dt.label.replace(/\s+/g, '-').toLowerCase()}`}
                            className="draft-type-btn"
                            onClick={() => handleDraftDocument(dt.label)}
                          >
                            <span>{dt.icon}</span>
                            <span>{dt.label}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {loading && (
            <div className="message-bubble ai">
              <div className="message-bubble-header">🤖 D Law</div>
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <textarea
            id="chat-input"
            className="chat-input"
            placeholder="Ask any Nigerian legal question (e.g. 'My employer fired me without notice. What are my rights?')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            id="send-btn"
            className="btn-send"
            onClick={handleSendMessage}
            disabled={!query.trim() || loading}
            title="Send (Enter)"
          >
            ✈️
          </button>
        </div>
      </div>

      {/* ── CITATION SOURCES PANEL ── */}
      <div className="references-panel">
        <div className="panel-header">📚 Citation Sources</div>
        <div className="panel-content">
          {references.length === 0 ? (
            <div className="panel-info">
              No references yet. Send a question to see the matching legal document sections here. Click any source to open its PDF page.
            </div>
          ) : (
            references.map((ref, idx) => (
              <div
                key={idx}
                id={`ref-card-${idx}`}
                className={`source-card ${activeRefIdx === idx ? 'active' : ''}`}
                onClick={() => handleOpenReference(ref, idx)}
              >
                <div className="source-card-header">
                  <span>📄 {ref.document_name}</span>
                  <span style={{ color: '#00ffaa' }}>Page {ref.page}</span>
                </div>
                <div className="source-card-body">{ref.page_content}</div>
                <div className="source-card-view-hint">👁️ Click to view in PDF</div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── PDF VIEWER PANEL ── */}
      {isPdfOpen && (
        <div className="pdf-viewer-panel">
          <div className="pdf-panel-header">
            <div className="pdf-panel-title">
              📄 Source Viewer
              <span className="pdf-panel-doc-name">{activePdfDocName}</span>
            </div>
            <div className="pdf-panel-controls">
              <span className="pdf-page-badge">Page {activePdfPage}</span>
              <button
                id="close-pdf-btn"
                className="btn-close-pdf"
                onClick={() => { setIsPdfOpen(false); setActiveRefIdx(null); }}
                title="Close PDF viewer"
              >
                ✕
              </button>
            </div>
          </div>
          <div className="pdf-iframe-container">
            {pdfLoading && (
              <div className="pdf-loading">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
                <span>Loading document...</span>
              </div>
            )}
            <iframe
              id="pdf-iframe"
              className="pdf-iframe"
              src={activePdfUrl}
              title={`${activePdfDocName} - Page ${activePdfPage}`}
              onLoad={() => setPdfLoading(false)}
              style={{ display: pdfLoading ? 'none' : 'block' }}
            />
          </div>
        </div>
      )}

      {/* ── DOCUMENT DRAFT MODAL ── */}
      {draftModal && (
        <div className="modal-overlay" onClick={(e) => { if (e.target.className === 'modal-overlay') setDraftModal(null); }}>
          <div className="modal-container" id="draft-modal">
            <div className="modal-header">
              <div className="modal-title-group">
                <h2 className="modal-title">
                  📄 {draftModal.document_type}
                  <span className="modal-badge">AI Generated</span>
                </h2>
                <div className="modal-subtitle">
                  Generated on {new Date(draftModal.generated_at).toLocaleString()} · Review before sending
                </div>
              </div>
              <div className="modal-header-actions">
                <button id="copy-doc-btn" className="btn-modal-action" onClick={handleCopyDocument}>
                  📋 Copy
                </button>
                <button id="download-doc-btn" className="btn-modal-action primary" onClick={handleDownloadDocument}>
                  ⬇️ Download .txt
                </button>
                <button id="close-modal-btn" className="btn-modal-close" onClick={() => setDraftModal(null)}>✕</button>
              </div>
            </div>

            <div className="modal-body">
              <div className="document-preview">{draftModal.document}</div>
            </div>

            <div className="modal-footer">
              <span>⚠️ Review and replace placeholder fields before sending this document.</span>
              {copySuccess && <span className="copy-success">✅ Copied to clipboard!</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
