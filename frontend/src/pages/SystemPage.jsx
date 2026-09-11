import { useEffect, useState } from 'react';
import {
  addResponseEngineer,
  deleteResponseEngineer,
  getSmtpSettings,
  listResponseEngineers,
  saveSmtpSettings,
  testSmtpSettings,
} from '../api/endpoints';

const SECTIONS = [
  { id: 'camera', nav: 'Camera & network', title: 'System', subtitle: 'Video camera list and upload settings' },
  { id: 'users', nav: 'User management', title: 'User management', subtitle: 'Accounts and active sessions' },
  { id: 'storage', nav: 'Storage', title: 'Storage', subtitle: 'Recording schedule and holiday rules' },
  { id: 'local', nav: 'Local', title: 'Local', subtitle: 'Engineers, departments, and Gmail SMTP' },
];

function LocalPanel() {
  const [engineers, setEngineers] = useState([]);
  const [name, setName] = useState('');
  const [department, setDepartment] = useState('');
  const [email, setEmail] = useState('');
  const [fromEmail, setFromEmail] = useState('');
  const [appPassword, setAppPassword] = useState('');
  const [smtpHost, setSmtpHost] = useState('smtp.gmail.com');
  const [smtpPort, setSmtpPort] = useState(587);
  const [hasPassword, setHasPassword] = useState(false);
  const [status, setStatus] = useState('');

  const load = async () => {
    const [eng, smtp] = await Promise.all([listResponseEngineers(), getSmtpSettings()]);
    setEngineers(eng.engineers || []);
    setFromEmail(smtp.from_email || '');
    setSmtpHost(smtp.smtp_host || 'smtp.gmail.com');
    setSmtpPort(smtp.smtp_port || 587);
    setHasPassword(Boolean(smtp.has_password));
  };

  useEffect(() => {
    load().catch((e) => setStatus(e.message || 'Failed to load Local settings'));
  }, []);

  const addEngineer = async () => {
    setStatus('');
    try {
      await addResponseEngineer({ name, department, email });
      setName('');
      setDepartment('');
      setEmail('');
      await load();
      setStatus('Engineer added.');
    } catch (e) {
      setStatus(e.message || 'Could not add engineer');
    }
  };

  const removeEngineer = async (id) => {
    if (!window.confirm('Remove this engineer?')) return;
    await deleteResponseEngineer(id);
    await load();
  };

  const saveMail = async () => {
    setStatus('');
    try {
      await saveSmtpSettings({
        from_email: fromEmail,
        app_password: appPassword,
        smtp_host: smtpHost,
        smtp_port: Number(smtpPort) || 587,
      });
      setAppPassword('');
      await load();
      setStatus('Gmail SMTP saved. Use Test email to confirm Google accepts the app password.');
    } catch (e) {
      setStatus(e.message || 'Could not save SMTP');
    }
  };

  const testMail = async () => {
    setStatus('Sending test email…');
    try {
      await testSmtpSettings({
        from_email: fromEmail,
        app_password: appPassword,
        smtp_host: smtpHost,
        smtp_port: Number(smtpPort) || 587,
      });
      setAppPassword('');
      await load();
      setStatus(`Test email sent to ${fromEmail}. Check inbox (and spam).`);
    } catch (e) {
      setStatus(e.message || 'Test email failed');
    }
  };

  return (
    <>
      <div className="glass-panel neu-card sys-panel">
        <h3>Engineers</h3>
        <p className="muted">Add name, department, and email. Alerts can assign a task and send mail to that person.</p>
        <div className="eng-form">
          <div className="field"><label>Name</label><input value={name} onChange={(e) => setName(e.target.value)} placeholder="Full name" /></div>
          <div className="field"><label>Department</label><input value={department} onChange={(e) => setDepartment(e.target.value)} placeholder="e.g. Lab Safety" /></div>
          <div className="field"><label>Email</label><input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@company.com" /></div>
          <button type="button" className="btn" onClick={addEngineer}>Add engineer</button>
        </div>
        {engineers.length === 0 ? (
          <p className="muted">No engineers yet.</p>
        ) : (
          <table className="data-table">
            <thead><tr><th>Name</th><th>Department</th><th>Email</th><th /></tr></thead>
            <tbody>
              {engineers.map((e) => (
                <tr key={e.id}>
                  <td>{e.name}</td>
                  <td>{e.department || '—'}</td>
                  <td>{e.email}</td>
                  <td><button type="button" className="link-btn" onClick={() => removeEngineer(e.id)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="glass-panel neu-card sys-panel" style={{ marginTop: 16 }}>
        <h3>Gmail SMTP</h3>
        <p className="muted">
          Send alert emails from your Gmail. You must use a Google <strong>App Password</strong>, not your normal Gmail password.
          Create one at Google Account → Security → 2-Step Verification → App passwords. Paste the 16 characters
          (spaces are removed automatically).
        </p>
        <div className="field"><label>From Gmail</label><input type="email" value={fromEmail} onChange={(e) => setFromEmail(e.target.value)} placeholder="you@gmail.com" /></div>
        <div className="field">
          <label>App password {hasPassword ? '(saved — paste a new one to replace)' : ''}</label>
          <input type="password" value={appPassword} onChange={(e) => setAppPassword(e.target.value)} placeholder={hasPassword ? 'Paste new app password to replace' : 'xxxx xxxx xxxx xxxx'} autoComplete="new-password" />
        </div>
        <div className="eng-form">
          <div className="field"><label>SMTP host</label><input value={smtpHost} onChange={(e) => setSmtpHost(e.target.value)} /></div>
          <div className="field"><label>Port</label><input value={smtpPort} onChange={(e) => setSmtpPort(e.target.value)} /></div>
        </div>
        <div className="eng-form">
          <button type="button" className="btn" onClick={saveMail}>Save SMTP</button>
          <button type="button" className="btn btn-secondary" onClick={testMail}>Send test email</button>
        </div>
      </div>
      {status && <p className="apply-status">{status}</p>}
    </>
  );
}

export default function SystemPage() {
  const [section, setSection] = useState('camera');
  const meta = SECTIONS.find((s) => s.id === section) || SECTIONS[0];

  return (
    <div className="system-page">
      <div className="sys-layout">
        <aside className="sys-sidebar glass-panel">
          <div className="sys-sidebar-title">Configuration</div>
          {SECTIONS.map((s) => (
            <button key={s.id} type="button" className={`sys-nav-btn ${section === s.id ? 'active' : ''}`} onClick={() => setSection(s.id)}>{s.nav}</button>
          ))}
        </aside>
        <div className="sys-main">
          <div className="sys-main-header">
            <h1 className="page-title gradient-text">{meta.title}</h1>
            <p className="page-subtitle">{meta.subtitle}</p>
          </div>
          {section === 'camera' && (
            <div className="glass-panel neu-card sys-panel">
              <h3>Demo Video Cameras</h3>
              <p className="muted">This demo uses uploaded MP4 files instead of RTSP streams. Videos loop seamlessly for all 6 camera feeds.</p>
              <div className="field"><label>Network Scan Subnet</label><input defaultValue="192.168.1.0/24" readOnly /></div>
              <button type="button" className="btn btn-secondary" disabled>Scan Network (demo)</button>
            </div>
          )}
          {section === 'users' && (
            <div className="glass-panel neu-card sys-panel">
              <table className="data-table">
                <thead><tr><th>Username</th><th>Level</th><th>Status</th></tr></thead>
                <tbody><tr><td>admin</td><td>admin</td><td><span className="badge ok">Active</span></td></tr></tbody>
              </table>
            </div>
          )}
          {section === 'storage' && (
            <div className="glass-panel neu-card sys-panel">
              <p className="muted">Recording schedules and holiday rules — demo UI.</p>
              <button type="button" className="btn btn-secondary" disabled>Add Schedule</button>
            </div>
          )}
          {section === 'local' && <LocalPanel />}
        </div>
      </div>
    </div>
  );
}
