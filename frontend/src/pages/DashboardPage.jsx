import { useEffect, useState } from 'react';
import { listCameras, liveStats, systemStats } from '../api/endpoints';

const TABS = [
  { id: 'all', label: 'Overview', key: null },
  { id: 'headcount', label: 'Head Count', key: 'headcount' },
  { id: 'entryexit', label: 'Entry / Exit', key: 'entryexit' },
  { id: 'flapgate', label: 'Flap Gate', key: 'flapgate' },
  { id: 'workforce', label: 'Workforce', key: 'workforce' },
  { id: 'intrusion', label: 'Intrusion', key: 'intrusion' },
  { id: 'ppe', label: 'PPE', key: 'ppe' },
  { id: 'firesmoke', label: 'Fire / Smoke', key: 'firesmoke' },
  { id: 'anpr', label: 'ANPR', key: 'anpr' },
  { id: 'fr', label: 'Face Rec', key: 'fr' },
  { id: 'beta', label: 'Beta AI', key: 'beta' },
  { id: 'automode', label: 'Autotrack', key: 'automode' },
];

const COLS = {
  headcount: [
    ['current_count', 'In view'],
    ['total_entries', 'Total'],
    ['status', 'Status'],
  ],
  entryexit: [
    ['entries', 'Entries'],
    ['exits', 'Exits'],
    ['current_inside', 'Inside'],
  ],
  flapgate: [
    ['trespassing_total', 'Trespass'],
    ['persons_in_frame', 'Persons'],
    ['status', 'Status'],
  ],
  workforce: [
    ['dwell_sec', 'Dwell s'],
    ['state', 'State'],
    ['alert', 'Alert'],
  ],
  intrusion: [
    ['detections', 'Detections'],
    ['alert', 'Alert'],
    ['status', 'Status'],
  ],
  ppe: [
    ['compliant_count', 'OK'],
    ['violation_count', 'Violations'],
    ['total_persons', 'People'],
  ],
  firesmoke: [
    ['fire_count', 'Fire'],
    ['smoke_count', 'Smoke'],
    ['status', 'Status'],
  ],
  anpr: [
    ['vehicles_detected', 'Vehicles'],
    ['plates_read', 'Plates'],
    ['overspeeding', 'Over speed'],
  ],
  fr: [
    ['faces_detected', 'Faces'],
    ['recognized', 'Known'],
    ['status', 'Status'],
  ],
  beta: [
    ['detections', 'Detections'],
    ['alerts', 'Alerts'],
    ['status', 'Status'],
  ],
  automode: [
    ['detections', 'Seen'],
    ['alerts', 'Alerts'],
    ['status', 'Status'],
  ],
};

function fmt(v) {
  if (v === true) return 'Yes';
  if (v === false) return 'No';
  if (v == null || v === '') return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(1);
  if (typeof v === 'object') return '—';
  return String(v);
}

function sumField(live, key) {
  return Object.values(live || {}).reduce((s, st) => s + (Number(st?.[key]) || 0), 0);
}

function headline(typeKey, live) {
  const n = Object.keys(live || {}).length;
  if (!n) return 'Off';
  switch (typeKey) {
    case 'headcount':
      return `${sumField(live, 'current_count')} in view`;
    case 'entryexit':
      return `${sumField(live, 'entries')} in · ${sumField(live, 'exits')} out`;
    case 'flapgate':
      return `${sumField(live, 'trespassing_total')} trespass`;
    case 'workforce':
      return `${n} station${n === 1 ? '' : 's'}`;
    case 'ppe':
      return `${sumField(live, 'violation_count')} violations`;
    case 'firesmoke':
      return `${sumField(live, 'fire_count') + sumField(live, 'smoke_count')} events`;
    case 'anpr':
      return `${sumField(live, 'plates_read')} plates`;
    case 'fr':
      return `${sumField(live, 'recognized')} recognized`;
    case 'automode':
    case 'beta':
      return `${sumField(live, 'alerts')} alerts`;
    default:
      return `${n} live`;
  }
}

function alertCount(alertsByType, key) {
  if (key === 'firesmoke') return alertsByType.firesmoke || alertsByType.fire_smoke || 0;
  return alertsByType[key] || 0;
}

export default function DashboardPage() {
  const [tab, setTab] = useState('all');
  const [cameras, setCameras] = useState([]);
  const [stats, setStats] = useState({});
  const [sys, setSys] = useState({});

  useEffect(() => {
    const load = async () => {
      const [c, s, sysS] = await Promise.all([listCameras(), liveStats(), systemStats()]);
      setCameras(Array.isArray(c) ? c : []);
      setStats(s || {});
      setSys(sysS || {});
    };
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  const stored = stats.stored || {};
  const alertsByType = stored.alerts_by_type || {};
  const active = stats.active_counts || {};
  const byId = Object.fromEntries(cameras.map((c) => [c.camera_id, c]));
  const types = TABS.filter((t) => t.key);
  const liveCount = types.reduce((s, t) => s + (active[t.key] || Object.keys(stats[t.key] || {}).length), 0);
  const current = TABS.find((t) => t.id === tab);

  return (
    <>
      <div className="page-header">
        <h1 className="page-title gradient-text">Analytics Dashboard</h1>
        <p className="page-subtitle">Live counts and stored totals for every AI detection</p>
      </div>

      <div className="dash-kpis">
        <div className="s-card glass-panel neu-card">
          <div className="s-label">Cameras</div>
          <div className="s-value">{sys.cameras_active ?? cameras.length}</div>
        </div>
        <div className="s-card glass-panel neu-card">
          <div className="s-label">Detections live</div>
          <div className="s-value accent">{liveCount}</div>
        </div>
        <div className="s-card glass-panel neu-card">
          <div className="s-label">Stored alerts</div>
          <div className="s-value">{stored.alerts_total ?? 0}</div>
        </div>
        <div className="s-card glass-panel neu-card">
          <div className="s-label">System</div>
          <div className="s-value s-value-sm">{Math.round(sys.cpu_percent || 0)}% · {Math.round(sys.memory_percent || 0)}%</div>
          <div className="s-sub">CPU · Memory</div>
        </div>
      </div>

      <div className="dash-tabs glass-panel">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === 'all' ? (
        <section className="dash-section glass-panel">
          <table className="data-table dash-table">
            <thead>
              <tr>
                <th>Detection</th>
                <th>Live cameras</th>
                <th>Now</th>
                <th>Stored alerts</th>
              </tr>
            </thead>
            <tbody>
              {types.map((t) => {
                const live = stats[t.key] || {};
                const n = active[t.key] || Object.keys(live).length;
                return (
                  <tr key={t.id} className="dash-row" onClick={() => setTab(t.id)}>
                    <td className="dash-type-name">{t.label}</td>
                    <td>{n ? `${n} live` : 'Off'}</td>
                    <td>{headline(t.key, live)}</td>
                    <td>{alertCount(alertsByType, t.key)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ) : (
        <section className="dash-section glass-panel">
          <h3>{current?.label}</h3>
          {Object.keys(stats[current?.key] || {}).length === 0 ? (
            <p className="muted">Not running. Enable it in AI Features{current?.key === 'automode' ? ' or start Autotrack' : ''}.</p>
          ) : (
            <table className="data-table dash-table">
              <thead>
                <tr>
                  <th>Camera</th>
                  {(COLS[current.key] || []).map(([k, label]) => <th key={k}>{label}</th>)}
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats[current.key] || {}).map(([cid, st]) => (
                  <tr key={cid}>
                    <td className="dash-type-name">{byId[cid]?.name || cid}</td>
                    {(COLS[current.key] || []).map(([k]) => (
                      <td key={k}>{fmt(st?.[k])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}
    </>
  );
}
