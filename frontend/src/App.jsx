import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('');
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

  const messagesEndRef = useRef(null);

  // Load API Key to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('GEMINI_API_KEY', apiKey);
  }, [apiKey]);

  // Fetch initial documents and sessions
  useEffect(() => {
    fetchDocuments();
    fetchSessions();
  }, []);

  // Fetch chat history whenever active session changes
  useEffect(() => {
    if (activeSession) {
      fetchSessionHistory(activeSession);
    }
  }, [activeSession]);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
        if (data.length > 0) {
          setSelectedDoc(data[0]);
        }
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
        // Clear references on switching chat
        setReferences([]);
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
        setErrorMsg(errData.detail || 'Failed to create chat session');
      }
    } catch (err) {
      setErrorMsg('Network error creating session.');
      console.error(err);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;
    if (!apiKey) {
      setErrorMsg('Please enter your Google Gemini API Key in the sidebar first.');
      return;
    }

    setErrorMsg('');
    const userQuery = query.trim();
    setQuery('');

    // Optimistically add user message to UI
    setMessages((prev) => [...prev, { sender: 'user', text: userQuery }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession,
          document_key: selectedDoc,
          language: language,
          query: userQuery,
          api_key: apiKey,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        // Add AI message to UI
        setMessages((prev) => [...prev, { sender: 'ai', text: data.answer }]);
        // Update references panel
        setReferences(data.references || []);
      } else {
        const errData = await res.json();
        setErrorMsg(errData.detail || 'An error occurred during communication.');
        // Remove optimistic user message on complete failure
        setMessages((prev) => prev.slice(0, -1));
      }
    } catch (err) {
      setErrorMsg('Network error communicating with backend.');
      console.error(err);
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

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">
            <span>⚖️</span> Super Legal
          </h1>

          <div className="settings-group">
            {errorMsg && <div className="alert-warning">{errorMsg}</div>}
            
            <label className="settings-label">Gemini API Key</label>
            <input
              type="password"
              className="text-input"
              placeholder="Paste your API key here..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />

            <label className="settings-label">Legal Document</label>
            <select
              className="select-input"
              value={selectedDoc}
              onChange={(e) => setSelectedDoc(e.target.value)}
            >
              {documents.map((doc) => (
                <option key={doc} value={doc}>
                  {doc}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* CHAT SESSIONS */}
        <div className="sessions-list-container">
          <div className="sessions-title">Chat Sessions</div>
          {sessions.map((sess) => (
            <div
              key={sess}
              className={`session-item ${activeSession === sess ? 'active' : ''}`}
              onClick={() => setActiveSession(sess)}
            >
              <span className="session-icon">💬</span>
              <span className="session-name">{sess}</span>
            </div>
          ))}
        </div>

        {/* CREATE NEW SESSION */}
        <form className="new-session-form" onSubmit={handleCreateSession}>
          <input
            type="text"
            className="text-input"
            placeholder="New chat name..."
            value={newSessionName}
            onChange={(e) => setNewSessionName(e.target.value)}
            style={{ marginBottom: '8px' }}
          />
          <button type="submit" className="btn-primary" style={{ width: '100%' }}>
            + Create New Chat
          </button>
        </form>
      </div>

      {/* CHAT AREA */}
      <div className="chat-area">
        <div className="chat-header">
          <div className="chat-header-info">
            <h2 className="chat-header-title">{activeSession}</h2>
            <div className="chat-header-subtitle">
              Querying: <span style={{ color: '#00ffaa' }}>{selectedDoc || 'None'}</span>
            </div>
          </div>

          <div className="chat-header-controls">
            <label className="settings-label" style={{ margin: 0 }}>Language:</label>
            <select
              className="select-input"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              style={{ width: '110px' }}
            >
              <option value="English">English</option>
              <option value="French">French</option>
              <option value="Yoruba">Yoruba</option>
              <option value="Hausa">Hausa</option>
              <option value="Igbo">Igbo</option>
            </select>
          </div>
        </div>

        {/* MESSAGES */}
        <div className="messages-container">
          {messages.length === 0 && !loading && (
            <div style={{ margin: 'auto', textAlign: 'center', opacity: 0.6, maxWidth: '400px' }}>
              <span style={{ fontSize: '3rem' }}>⚖️</span>
              <h3>Welcome to Super Legal</h3>
              <p style={{ fontSize: '0.9rem' }}>
                Ask any question about Nigerian Law. Choose the correct document in the sidebar to retrieve facts, and choose your preferred language above.
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message-bubble ${msg.sender}`}>
              <div className="message-bubble-header">
                {msg.sender === 'user' ? '🧑‍⚖️ Client' : '🤖 Assistant'}
              </div>
              <div className="message-text">{msg.text}</div>
            </div>
          ))}

          {loading && (
            <div className="message-bubble ai">
              <div className="message-bubble-header">🤖 Assistant</div>
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* INPUT */}
        <div className="chat-input-container">
          <textarea
            className="chat-input"
            placeholder="Ask a legal question (e.g. 'What is the minimum wage under the Labour Act?')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="btn-send"
            onClick={handleSendMessage}
            disabled={!query.trim() || loading}
          >
            ✈️
          </button>
        </div>
      </div>

      {/* REFERENCES PANEL */}
      <div className="references-panel">
        <div className="panel-header">📚 Citation Sources</div>
        <div className="panel-content">
          {references.length === 0 ? (
            <div className="panel-info">
              No references retrieved yet. Send a question to see the matching legal document sections here.
            </div>
          ) : (
            references.map((ref, idx) => (
              <div key={idx} className="source-card">
                <div className="source-card-header">
                  <span>Source #{idx + 1}</span>
                  <span style={{ color: '#00ffaa' }}>
                    Page {ref.metadata?.page !== undefined ? ref.metadata.page + 1 : 'N/A'}
                  </span>
                </div>
                <div className="source-card-body">{ref.page_content}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
