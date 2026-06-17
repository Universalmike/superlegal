import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const DOCUMENT_TYPES = [
  { icon: '📜', label: 'Demand Letter' },
  { icon: '🏠', label: 'Quit Notice' },
  { icon: '⚖️', label: 'Formal Complaint Letter' },
  { icon: '📋', label: 'Employment Rights Notice' },
  { icon: '📝', label: 'Statement of Facts' },
];

const NIGERIAN_STATES = [
  'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno',
  'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'FCT (Abuja)', 'Gombe',
  'Imo', 'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara',
  'Lagos', 'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau',
  'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara',
];

const NBA_BRANCHES = {
  'FCT (Abuja)': 'NBA Abuja Branch — Plot 1058 Nkwere Street, Area 3, Garki',
  'Lagos': 'NBA Lagos Branch — 1 Tinubu Square, Lagos Island',
  'Rivers': 'NBA Port Harcourt Branch — 36 Aggrey Road, Port Harcourt',
  'Kano': 'NBA Kano Branch — No. 3 Hospital Road, Kano',
  'Oyo': 'NBA Ibadan Branch — Dugbe, Ring Road, Ibadan',
  'Enugu': 'NBA Enugu Branch — GRA, Enugu',
  'Anambra': 'NBA Awka/Onitsha Branch — Awka, Anambra State',
  'Delta': 'NBA Asaba/Warri Branch — Asaba, Delta State',
  'Edo': 'NBA Benin City Branch — 1 Lawani Street, Benin City',
  'Kaduna': 'NBA Kaduna Branch — 5 Ahmadu Bello Way, Kaduna',
  'Imo': 'NBA Owerri Branch — Douglas Road, Owerri',
  'Abia': 'NBA Umuahia/Aba Branch — Aba Road, Umuahia',
  'Benue': 'NBA Makurdi Branch — High Level, Makurdi',
  'Cross River': 'NBA Calabar Branch — 10 Murtala Mohammed Way, Calabar',
  'Akwa Ibom': 'NBA Uyo Branch — 3 Abak Road, Uyo',
  'Kwara': 'NBA Ilorin Branch — Fate Road, Ilorin',
  'Ogun': 'NBA Abeokuta Branch — Oke-Lantoro, Abeokuta',
  'Ondo': 'NBA Akure Branch — Alagbaka Estate, Akure',
  'Osun': 'NBA Osogbo Branch — Oke-Fia, Osogbo',
  'Ekiti': 'NBA Ado-Ekiti Branch — Ajilosun, Ado-Ekiti',
  'Niger': 'NBA Minna Branch — Paiko Road, Minna',
  'Kebbi': 'NBA Birnin-Kebbi Branch — Birnin-Kebbi, Kebbi State',
  'Sokoto': 'NBA Sokoto Branch — Maiduguri Road, Sokoto',
  'Zamfara': 'NBA Gusau Branch — Gusau, Zamfara State',
  'Katsina': 'NBA Katsina Branch — Katsina, Katsina State',
  'Jigawa': 'NBA Dutse Branch — Dutse, Jigawa State',
  'Bauchi': 'NBA Bauchi Branch — Jos Road, Bauchi',
  'Gombe': 'NBA Gombe Branch — Tudun Wada, Gombe',
  'Yobe': 'NBA Damaturu Branch — Damaturu, Yobe State',
  'Borno': 'NBA Maiduguri Branch — Maiduguri, Borno State',
  'Adamawa': 'NBA Yola Branch — Yola, Adamawa State',
  'Taraba': 'NBA Jalingo Branch — Jalingo, Taraba State',
  'Plateau': 'NBA Jos Branch — Tudun Wada, Jos',
  'Nasarawa': 'NBA Lafia Branch — Lafia, Nasarawa State',
  'Kogi': 'NBA Lokoja Branch — Lokoja, Kogi State',
  'Bayelsa': 'NBA Yenagoa Branch — Yenagoa, Bayelsa State',
  'Ebonyi': 'NBA Abakaliki Branch — Abakaliki, Ebonyi State',
};

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('All Documents');
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState('Chat 1');
  const [messages, setMessages] = useState([]);
  const [language, setLanguage] = useState('English');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
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

  // Find a Lawyer state
  const [lawyerModal, setLawyerModal] = useState(null); // null | { violationText: string|null }
  const [selectedState, setSelectedState] = useState('');

  // Consultation mode state
  const [consultMode, setConsultMode] = useState(false);
  const [consultStage, setConsultStage] = useState('issue'); // 'issue' | 'questions' | 'assessing'
  const [consultIssue, setConsultIssue] = useState('');
  const [consultQuestions, setConsultQuestions] = useState([]);
  const [consultAnswers, setConsultAnswers] = useState([]);
  const [consultCurrentQ, setConsultCurrentQ] = useState(0);
  const [consultInput, setConsultInput] = useState('');
  const [consultLoading, setConsultLoading] = useState(false);

  const messagesEndRef = useRef(null);
  const popoverRef = useRef(null);
  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [speakingIdx, setSpeakingIdx] = useState(null);

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
      window.speechSynthesis?.cancel();
      setSpeakingIdx(null);
      recognitionRef.current?.stop();
      setIsListening(false);
    }
  }, [activeSession]);

  useEffect(() => {
    window.speechSynthesis?.cancel();
    setSpeakingIdx(null);
  }, [language]);

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
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const vDetected = detectViolation(data.answer);
        setMessages((prev) => [...prev, {
          sender: 'ai',
          text: data.answer,
          violationDetected: vDetected,
          violationText: vDetected ? extractViolationSentence(data.answer) : null,
        }]);
        setReferences(data.references || []);
      } else {
        let msg = 'An error occurred.';
        try { const e = await res.json(); msg = e.detail || msg; } catch {}
        setErrorMsg(msg);
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
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setDraftModal(data);
      } else {
        let msg = 'Document generation failed.';
        try { const e = await res.json(); msg = e.detail || msg; } catch {}
        setErrorMsg(msg);
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

  const getLangCode = (lang) => {
    const codes = { English: 'en-NG', Pidgin: 'en-NG', Yoruba: 'yo', Hausa: 'ha', Igbo: 'ig', French: 'fr-FR' };
    return codes[lang] || 'en-NG';
  };

  const detectViolation = (text) => {
    const patterns = [
      /\bbreached?\b/i, /\bviolat(ed|ion|ing)\b/i,
      /\bunlawful(ly)?\b/i, /\billegal(ly)?\b/i, /\bwrongful(ly)?\b/i,
      /\bcontravene[sd]?\b/i, /\binfring(ed|es|ing)\b/i,
      /\bentitled to (compensation|damages|remedy|reinstatement)\b/i,
      /\bwrongful (termination|dismissal)\b/i, /\bunfair dismissal\b/i,
      /\byour (rights|employer) (has|have|had|was|were) (violated|breached|infringed)\b/i,
    ];
    return patterns.some((p) => p.test(text));
  };

  const extractViolationSentence = (text) => {
    const clean = text.replace(/\*\*/g, '');
    const sentences = clean.split(/(?<=[.!?])\s+/);
    const keyword = /breach|violat|unlawful|illegal|wrongful|contravene|infringe|entitled to compensation/i;
    const found = sentences.find((s) => keyword.test(s));
    return found ? found.trim().slice(0, 240) : null;
  };

  const handleToggleListen = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setErrorMsg('Voice input requires Chrome or Edge browser.');
      return;
    }
    window.speechSynthesis?.cancel();
    setSpeakingIdx(null);
    const recognition = new SpeechRecognition();
    recognition.lang = getLangCode(language);
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (e) => {
      const transcript = Array.from(e.results).map((r) => r[0].transcript).join('');
      setQuery(transcript);
    };
    recognition.onend = () => setIsListening(false);
    recognition.onerror = (e) => {
      setIsListening(false);
      if (e.error !== 'no-speech') setErrorMsg(`Mic error: ${e.error}. Please try again.`);
    };
    recognitionRef.current = recognition;
    recognition.start();
  };

  const handleSpeak = (text, idx) => {
    if (!('speechSynthesis' in window)) {
      setErrorMsg('Audio playback is not supported in this browser.');
      return;
    }
    window.speechSynthesis.cancel();
    if (speakingIdx === idx) { setSpeakingIdx(null); return; }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = getLangCode(language);
    utterance.rate = 0.9;
    utterance.onend = () => setSpeakingIdx(null);
    utterance.onerror = () => setSpeakingIdx(null);
    setSpeakingIdx(idx);
    window.speechSynthesis.speak(utterance);
  };

  const handleStartConsultation = () => {
    window.speechSynthesis?.cancel();
    setSpeakingIdx(null);
    recognitionRef.current?.stop();
    setIsListening(false);
    setQuery('');
    setConsultMode(true);
    setConsultStage('issue');
    setConsultIssue('');
    setConsultQuestions([]);
    setConsultAnswers([]);
    setConsultCurrentQ(0);
    setConsultInput('');
    setMessages((prev) => [
      ...prev,
      { sender: 'ai', text: '🧑‍⚖️ Welcome to Consultation Mode.\n\nPlease describe your legal situation briefly — for example: "My employer fired me without notice" or "My landlord refuses to return my deposit." I will then ask you 3 targeted questions before giving a full legal assessment.' },
    ]);
  };

  const handleConsultIssueSubmit = async () => {
    if (!consultInput.trim() || consultLoading) return;
    const issue = consultInput.trim();
    setConsultIssue(issue);
    setConsultInput('');
    setMessages((prev) => [...prev, { sender: 'user', text: issue }]);
    setConsultLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/consultation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSession, issue, document_key: selectedDoc }),
      });
      if (res.ok) {
        const data = await res.json();
        setConsultQuestions(data.questions);
        setConsultCurrentQ(0);
        setConsultStage('questions');
        setMessages((prev) => [
          ...prev,
          { sender: 'ai', text: `Thank you. I have 3 targeted questions for you.\n\n**Question 1 of ${data.questions.length}:** ${data.questions[0]}` },
        ]);
      } else {
        let msg = 'Failed to start consultation.';
        try { const e = await res.json(); msg = e.detail || msg; } catch {}
        setErrorMsg(msg);
        setConsultMode(false);
      }
    } catch {
      setErrorMsg('Network error starting consultation.');
      setConsultMode(false);
    } finally {
      setConsultLoading(false);
    }
  };

  const handleConsultAnswer = async () => {
    if (!consultInput.trim() || consultLoading) return;
    const answer = consultInput.trim();
    const newAnswers = [...consultAnswers, answer];
    setConsultAnswers(newAnswers);
    setConsultInput('');
    setMessages((prev) => [...prev, { sender: 'user', text: answer }]);

    const nextQ = consultCurrentQ + 1;

    if (nextQ < consultQuestions.length) {
      setConsultCurrentQ(nextQ);
      setMessages((prev) => [
        ...prev,
        { sender: 'ai', text: `**Question ${nextQ + 1} of ${consultQuestions.length}:** ${consultQuestions[nextQ]}` },
      ]);
    } else {
      setConsultStage('assessing');
      setConsultLoading(true);
      setMessages((prev) => [
        ...prev,
        { sender: 'ai', text: '⚖️ Thank you. Analyzing your situation against Nigerian law — this may take a moment...' },
      ]);
      try {
        const res = await fetch(`${API_BASE}/api/consultation/assess`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: activeSession,
            issue: consultIssue,
            questions: consultQuestions,
            answers: newAnswers,
            document_key: selectedDoc,
            language,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setMessages((prev) => [
            ...prev.slice(0, -1),
            {
              sender: 'ai',
              text: data.assessment,
              isConsultation: true,
              violationDetected: true,
              violationText: extractViolationSentence(data.assessment),
            },
          ]);
          setReferences(data.references || []);
        } else {
          let msg = 'Assessment failed.';
          try { const e = await res.json(); msg = e.detail || msg; } catch {}
          setErrorMsg(msg);
        }
      } catch {
        setErrorMsg('Network error during assessment.');
      } finally {
        setConsultLoading(false);
        setConsultMode(false);
        setConsultStage('issue');
      }
    }
  };

  const handleExitConsultation = () => {
    setConsultMode(false);
    setConsultStage('issue');
    setConsultInput('');
    setConsultLoading(false);
  };

  return (
    <div className="app-container">
      {/* ── SIDEBAR ── */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">⚖️ Super Legal</h1>
          <div className="settings-group">
            {errorMsg && <div className="alert-warning">{errorMsg}</div>}
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
            <button
              className={`btn-consult-mode${consultMode ? ' active' : ''}`}
              onClick={consultMode ? handleExitConsultation : handleStartConsultation}
              disabled={loading}
              title="Simulate a real lawyer client intake interview"
            >
              {consultMode ? '✕ Exit Consultation' : '🧑‍⚖️ Consultation Mode'}
            </button>
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
              <div key={index} className={`message-bubble ${msg.sender}${msg.isConsultation ? ' consultation-result' : ''}`}>
                <div className="message-bubble-header">
                  {msg.sender === 'user' ? '🧑‍⚖️ Client' : msg.isConsultation ? '⚖️ Legal Assessment' : '🤖 D Law'}
                  {msg.isConsultation && <span className="consult-result-badge">Consultation Result</span>}
                </div>
                <div className="message-text">{msg.text}</div>

                {/* Draft action row — only shown on AI messages */}
                {msg.sender === 'ai' && (
                  <div className="message-actions">
                    <button
                      className={`btn-speak${speakingIdx === index ? ' speaking' : ''}`}
                      onClick={() => handleSpeak(msg.text, index)}
                      title={speakingIdx === index ? 'Stop audio' : 'Listen to this response'}
                    >
                      {speakingIdx === index ? '⏹ Stop' : '🔊 Listen'}
                    </button>
                    {msg.violationDetected && (
                      <button
                        className="btn-legal-help"
                        onClick={() => { setSelectedState(''); setLawyerModal({ violationText: msg.violationText }); }}
                        title="Connect with legal aid or a verified Nigerian lawyer"
                      >
                        ⚖️ Find a Lawyer
                      </button>
                    )}
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

        {consultMode ? (
          <div className="consultation-input-wrapper">
            <div className="consultation-status-bar">
              <div className="consult-status-left">
                <span className="consult-mode-badge">🧑‍⚖️ Consultation Mode</span>
                {consultStage === 'questions' && (
                  <>
                    <div className="consult-progress-dots">
                      {consultQuestions.map((_, i) => (
                        <span
                          key={i}
                          className={`consult-dot${i < consultCurrentQ ? ' done' : i === consultCurrentQ ? ' active' : ''}`}
                        />
                      ))}
                    </div>
                    <span className="consult-step-label">Question {consultCurrentQ + 1} of {consultQuestions.length}</span>
                  </>
                )}
                {consultStage === 'assessing' && <span className="consult-step-label">⚖️ Preparing assessment...</span>}
              </div>
              <button className="btn-exit-consult" onClick={handleExitConsultation} title="Exit consultation mode">
                ✕ Exit
              </button>
            </div>
            <div className="chat-input-container">
              <textarea
                className="chat-input"
                placeholder={
                  consultStage === 'issue'
                    ? 'Briefly describe your legal situation (e.g. "My employer fired me without notice")...'
                    : 'Type your answer here...'
                }
                value={consultInput}
                onChange={(e) => setConsultInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    consultStage === 'issue' ? handleConsultIssueSubmit() : handleConsultAnswer();
                  }
                }}
                disabled={consultLoading || consultStage === 'assessing'}
              />
              <button
                className="btn-send"
                onClick={consultStage === 'issue' ? handleConsultIssueSubmit : handleConsultAnswer}
                disabled={!consultInput.trim() || consultLoading || consultStage === 'assessing'}
                title={consultStage === 'issue' ? 'Begin consultation (Enter)' : 'Submit answer (Enter)'}
              >
                {consultLoading || consultStage === 'assessing' ? '⏳' : '→'}
              </button>
            </div>
          </div>
        ) : (
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
              type="button"
              className={`btn-mic${isListening ? ' listening' : ''}`}
              onClick={handleToggleListen}
              disabled={loading}
              title={isListening ? 'Stop listening' : 'Voice input — click to speak'}
            >
              {isListening ? '⏹' : '🎤'}
            </button>
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
        )}
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
      {/* ── FIND A LAWYER MODAL ── */}
      {lawyerModal && (
        <div className="modal-overlay" onClick={(e) => { if (e.target.className === 'modal-overlay') setLawyerModal(null); }}>
          <div className="modal-container lawyer-modal" id="lawyer-modal">
            <div className="modal-header">
              <div className="modal-title-group">
                <h2 className="modal-title">⚖️ Find Legal Help</h2>
                <div className="modal-subtitle">Connect with free legal aid or a verified Nigerian lawyer</div>
              </div>
              <button className="btn-modal-close" onClick={() => setLawyerModal(null)}>✕</button>
            </div>

            <div className="modal-body lawyer-modal-body">
              {lawyerModal.violationText && (
                <div className="violation-banner">
                  <span className="violation-banner-icon">⚠️</span>
                  <div>
                    <div className="violation-banner-label">Potential legal issue detected</div>
                    <div className="violation-banner-text">"{lawyerModal.violationText}"</div>
                  </div>
                </div>
              )}

              <div className="lawyer-section-title">🏛️ Free Legal Aid — Available Nationwide</div>
              <div className="resource-cards">
                <div className="resource-card">
                  <div className="resource-card-header">
                    <span className="resource-card-name">Legal Aid Council of Nigeria (LACON)</span>
                    <span className="resource-card-badge free">Free</span>
                  </div>
                  <div className="resource-card-desc">Government-funded body providing free legal services to Nigerians who cannot afford a lawyer. Has offices in all 36 states.</div>
                  <div className="resource-card-links">
                    <a className="resource-link" href="https://legalaidcouncil.gov.ng" target="_blank" rel="noopener noreferrer">🌐 Website</a>
                    <a className="resource-link" href="tel:+2349080000000">📞 Head Office</a>
                  </div>
                </div>

                <div className="resource-card">
                  <div className="resource-card-header">
                    <span className="resource-card-name">National Human Rights Commission (NHRC)</span>
                    <span className="resource-card-badge free">Free</span>
                  </div>
                  <div className="resource-card-desc">Handles complaints about human rights and constitutional violations. Can compel government agencies and employers to comply.</div>
                  <div className="resource-card-links">
                    <a className="resource-link" href="https://nhrc-ng.org" target="_blank" rel="noopener noreferrer">🌐 Website</a>
                    <span className="resource-link-text">📍 26 Aguiyi-Ironsi St, Maitama, Abuja</span>
                  </div>
                </div>

                <div className="resource-card">
                  <div className="resource-card-header">
                    <span className="resource-card-name">Federation of Women Lawyers — FIDA Nigeria</span>
                    <span className="resource-card-badge">Gender &amp; Family</span>
                  </div>
                  <div className="resource-card-desc">Provides free legal services for women's rights, domestic issues, and gender-based discrimination under Nigerian law.</div>
                  <div className="resource-card-links">
                    <a className="resource-link" href="https://fidanigeria.org" target="_blank" rel="noopener noreferrer">🌐 Website</a>
                  </div>
                </div>
              </div>

              <div className="lawyer-section-title" style={{ marginTop: '24px' }}>📍 Find a Private Lawyer by State</div>
              <div className="state-selector-row">
                <select
                  className="select-input lawyer-state-select"
                  value={selectedState}
                  onChange={(e) => setSelectedState(e.target.value)}
                >
                  <option value="">Select your state...</option>
                  {NIGERIAN_STATES.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>

              {selectedState && (
                <div className="nba-result-card">
                  <div className="nba-result-header">
                    <span>🏛️ Nigerian Bar Association (NBA)</span>
                    <span className="resource-card-badge">Verified</span>
                  </div>
                  <div className="nba-branch-name">
                    {NBA_BRANCHES[selectedState] || `NBA ${selectedState} Branch`}
                  </div>
                  <div className="nba-result-desc">
                    The NBA {selectedState} Branch can refer you to qualified lawyers practicing in {selectedState}. Visit the NBA national portal to search by specialisation (employment, tenancy, criminal).
                  </div>
                  <div className="resource-card-links" style={{ marginTop: '12px' }}>
                    <a className="resource-link primary" href="https://www.nigerianbar.org.ng" target="_blank" rel="noopener noreferrer">🌐 NBA National Portal</a>
                  </div>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <span>⚠️ Super Legal provides legal information, not legal advice. Consult a qualified Nigerian lawyer for your specific situation.</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
