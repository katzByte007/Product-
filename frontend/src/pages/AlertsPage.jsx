import { useCallback, useEffect, useState } from 'react';
import { deleteAlert, deleteAllAlerts, assignAlert, listAlerts, listResponseEngineers } from '../api/endpoints';

const FILTERS = [
  { value: '', label: 'All' },
  { value: 'automode', label: 'Autotrack' },
  { value: 'beta_owlv2', label: 'Beta OWLv2' },
  { value: 'beta_qwen_anomaly', label: 'Beta Qwen' },
  { value: 'ppe', label: 'PPE' },
  { value: 'fire_smoke', label: 'Fire/Smoke' },
  { value: 'workforce', label: 'Workforce' },
  { value: 'intrusion', label: 'Intrusion' },
  { value: 'no_ppe', label: 'No PPE' },
];

export default function AlertsPage() {
  const [filter, setFilter] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [engineers, setEngineers] = useState([]);
  const [picked, setPicked] = useState({});
  const [busyId, setBusyId] = useState(null);
  const [notice, setNotice] = useState('');

  const load = useCallback(async () => {
    const [data, eng] = await Promise.all([listAlerts(filter), listResponseEngineers()]);
    setAlerts(data.alerts || []);
    setEngineers(eng.engineers || []);
  }, [filter]);

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [load]);

  const onAssign = async (alertId) => {
    const engineerId = Number(picked[alertId]);
    if (!engineerId) {
      setNotice('Select an engineer first. Add people in System → Local.');
      return;
    }
    setBusyId(alertId);
    setNotice('');
    try {
      const res = await assignAlert(alertId, engineerId);
      if (res.email_sent) {
        setNotice('Assigned and emailed the engineer.');
      } else {
        setNotice(res.email_error || 'Assigned, but email was not sent. Check Gmail settings in System → Local.');
      }
      await load();
    } catch (e) {
      setNotice(e.message || 'Assign failed');
    } finally {
      setBusyId(null);
    }
  };

  const onDelete = async (alertId) => {
    if (!window.confirm('Delete this alert?')) return;
    setBusyId(alertId);
    try {
      await deleteAlert(alertId);
      setAlerts((prev) => prev.filter((a) => a.id !== alertId));
    } catch (e) {
      setNotice(e.message || 'Delete failed');
    } finally {
      setBusyId(null);
    }
  };

  const onDeleteAll = async () => {
    if (!window.confirm('Delete ALL generated alerts? This cannot be undone.')) return;
    setBusyId('all');
    try {
      await deleteAllAlerts();
      setAlerts([]);
      setNotice('All alerts deleted.');
    } catch (e) {
      setNotice(e.message || 'Delete all failed');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <>
      <div className="page-header">
        <h1 className="page-title gradient-text">Alerts</h1>
        <p className="page-subtitle">Snapshot evidence from AI detections and Autotrack. Assign an engineer to email the alert from your Gmail.</p>
      </div>
      <div className="filter-bar glass-panel">
        {FILTERS.map((f) => (
          <button key={f.value} type="button" className={filter === f.value ? 'active' : ''} onClick={() => setFilter(f.value)}>{f.label}</button>
        ))}
        {alerts.length > 0 && (
          <button type="button" className="btn btn-sm btn-danger-ghost" disabled={busyId === 'all'} onClick={onDeleteAll}>
            Delete all
          </button>
        )}
      </div>
      {notice && <p className="apply-status">{notice}</p>}
      {alerts.length === 0 ? (
        <div className="empty-state glass-panel">
          No alerts yet — enable Autotrack (top-right) or configure AI Features to start monitoring.
        </div>
      ) : (
        <div className="alert-list">
          {alerts.map((a) => (
            <div key={a.id} className="alert-card glass-panel neu-card">
              {a.snapshot_url && (
                <div className="alert-snapshot-wrap">
                  <img src={a.snapshot_url} alt={`Alert ${a.id}`} className="alert-snapshot" loading="lazy" />
                </div>
              )}
              <div className="alert-body">
                <div className="alert-meta">
                  <span className="badge">{a.detection_type === 'automode' ? 'autotrack' : a.detection_type}</span>
                  <span className="muted">{a.camera_id}</span>
                  <span className={`severity ${a.severity}`}>{a.severity}</span>
                  {a.created_at && <span className="muted">{a.created_at}</span>}
                </div>
                <p className="alert-message">{a.message}</p>
                <div className="alert-actions">
                  <select
                    value={picked[a.id] || ''}
                    onChange={(e) => setPicked((prev) => ({ ...prev, [a.id]: e.target.value }))}
                  >
                    <option value="">Assign engineer…</option>
                    {engineers.map((e) => (
                      <option key={e.id} value={e.id}>{e.name}{e.department ? ` · ${e.department}` : ''}</option>
                    ))}
                  </select>
                  <button type="button" className="btn btn-sm" disabled={busyId === a.id} onClick={() => onAssign(a.id)}>
                    {busyId === a.id ? 'Sending…' : 'Assign & email'}
                  </button>
                  <button type="button" className="btn btn-sm btn-danger-ghost" disabled={busyId === a.id} onClick={() => onDelete(a.id)}>
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
