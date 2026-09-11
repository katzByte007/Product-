import { useMemo, useState } from 'react';
import {
  DETECTION_CATEGORIES,
  CAMERA_TIERS,
  filterDetectionTypes,
} from './detectionTypes';

const TIER_BADGE = {
  standard: { cls: 'tier-standard', label: 'Standard camera' },
  high_resolution: { cls: 'tier-hires', label: '1080p+ recommended' },
  dedicated: { cls: 'tier-dedicated', label: 'Dedicated camera' },
};

export default function AiTypeCards({ stats, onSelect }) {
  const [category, setCategory] = useState('all');
  const [cameraTier, setCameraTier] = useState('all');
  const [worksNowOnly, setWorksNowOnly] = useState(false);

  const activeMap = {
    headcount: stats.headcount, entryexit: stats.entryexit, flapgate: stats.flapgate,
    workforce: stats.workforce, intrusion: stats.intrusion,
    ppe: stats.ppe, firesmoke: stats.firesmoke, anpr: stats.anpr, beta: stats.beta, fr: stats.fr,
  };

  const filtered = useMemo(
    () => filterDetectionTypes({ category, cameraTier, standardCamerasOnly: worksNowOnly }),
    [category, cameraTier, worksNowOnly],
  );

  return (
    <>
      <div className="section-title gradient-text">AI Features</div>
      <p className="muted" style={{ fontSize: '0.85em', marginBottom: 12 }}>
        Select a detection module. Use the filters to match camera hardware.
      </p>

      <div className="ai-filter-bar glass-panel">
        <label className="filter-select">
          <span>Category</span>
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            {DETECTION_CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>{c.label}</option>
            ))}
          </select>
        </label>
        <label className="filter-select">
          <span>Camera requirement</span>
          <select value={cameraTier} onChange={(e) => setCameraTier(e.target.value)}>
            {CAMERA_TIERS.map((t) => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </select>
        </label>
        <label className="works-now-toggle">
          <input type="checkbox" checked={worksNowOnly} onChange={(e) => setWorksNowOnly(e.target.checked)} />
          Compatible with current cameras
        </label>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state glass-panel">No features match the selected filters.</div>
      ) : (
        <div className="type-cards">
          {filtered.map((t) => {
            const tier = TIER_BADGE[t.cameraTier] || TIER_BADGE.standard;
            const catLabel = DETECTION_CATEGORIES.find((c) => c.id === t.category)?.label || t.category;
            return (
              <button key={t.id} type="button" className={`type-card glass-panel neu-card ${t.cardClass}`} onClick={() => onSelect(t.id)}>
                <div className="tc-badges">
                  <span className={`tier-badge ${tier.cls}`}>{t.cameraTierLabel || tier.label}</span>
                  <span className="cat-badge">{catLabel}</span>
                </div>
                <div className="tc-icon">{t.icon}</div>
                <h3>{t.title}</h3>
                <p>{t.description}</p>
                {t.dedicatedNote ? <p className="tc-note">{t.dedicatedNote}</p> : null}
                {t.minResolution ? <p className="tc-res">Min resolution: {t.minResolution}</p> : null}
                <div className="tc-stat">
                  <div className="tc-stat-item">
                    <div className="num">{activeMap[t.id] ?? 0}</div>
                    <div className="lbl">Active</div>
                  </div>
                  <div className="tc-stat-item">
                    <div className="num">{stats.total}</div>
                    <div className="lbl">Total Cameras</div>
                  </div>
                </div>
                <div className="tc-arrow">→</div>
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}
