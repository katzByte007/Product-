import { useCallback, useEffect, useState } from 'react';
import { getAutoModeConfig, listCameras, saveAutoModeConfig } from '../api/endpoints';
import ScheduleFields from './ai-config/ScheduleFields';

const DEFAULT_PROMPT = 'person';

function cameraList(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.cameras)) return data.cameras;
  return [];
}

export default function AutoModePanel({ open, onClose }) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [cameraIds, setCameraIds] = useState([]);
  const [enabled, setEnabled] = useState(false);
  const [confidence, setConfidence] = useState(0.2);
  const [model, setModel] = useState('owlv2');
  const [cameras, setCameras] = useState([]);
  const [activeCameras, setActiveCameras] = useState([]);
  const [owlAvailable, setOwlAvailable] = useState(true);
  const [yoloWorldAvailable, setYoloWorldAvailable] = useState(false);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [scheduleEnabled, setScheduleEnabled] = useState(false);
  const [scheduleStart, setScheduleStart] = useState('08:00');
  const [scheduleEnd, setScheduleEnd] = useState('18:00');

  const load = useCallback(async () => {
    try {
      const [cfg, camData] = await Promise.all([getAutoModeConfig(), listCameras()]);
      setPrompt(cfg.prompt || DEFAULT_PROMPT);
      setCameraIds(cfg.camera_ids || []);
      setEnabled(Boolean(cfg.enabled));
      setConfidence(cfg.confidence ?? 0.2);
      setModel(cfg.model || 'owlv2');
      setActiveCameras(cfg.active_cameras || []);
      setOwlAvailable(cfg.owlv2_available !== false);
      setYoloWorldAvailable(Boolean(cfg.yolo_world_available));
      setScheduleEnabled(Boolean(cfg.schedule_enabled));
      setScheduleStart(cfg.schedule_start || '08:00');
      setScheduleEnd(cfg.schedule_end || '18:00');
      setCameras(cameraList(camData));
    } catch (e) {
      setStatus(e.message || 'Failed to load Autotrack config');
    }
  }, []);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  const toggleCamera = (id) => {
    setCameraIds((prev) => (prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]));
  };

  const handleApply = async () => {
    setLoading(true);
    setStatus('');
    try {
      const res = await saveAutoModeConfig({
        enabled,
        prompt,
        camera_ids: cameraIds,
        confidence,
        model,
        schedule_enabled: scheduleEnabled,
        schedule_start: scheduleStart,
        schedule_end: scheduleEnd,
      });
      setActiveCameras(res.active_cameras || []);
      if (res.model) setModel(res.model);
      if (res.prompt) setPrompt(res.prompt);
      if (res.scheduled_idle) {
        setStatus('Saved. Autotrack will start during the scheduled hours.');
      } else if (res.errors?.length) {
        setStatus(`Started on ${(res.active_cameras || []).length} camera(s). Issues: ${res.errors.join('; ')}`);
      } else if (enabled) {
        const modelName = model === 'yolo-world' ? 'YOLO-World' : 'OWLv2';
        setStatus(`Autotrack (${modelName}) watching ${(res.active_cameras || []).length} camera(s) for: ${res.prompt || prompt}. Alerts appear after continuous detections.`);
      } else {
        setStatus('Autotrack disabled.');
      }
    } catch (e) {
      setStatus(e.message || 'Failed to apply Autotrack');
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;
  const modelAvailable = model === 'yolo-world' ? yoloWorldAvailable : owlAvailable;
  const modelName = model === 'yolo-world' ? 'YOLO-World' : 'OWLv2';

  return (
    <div className="automode-overlay" role="dialog" aria-modal="true" aria-label="Autotrack">
      <div className="automode-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="automode-panel glass-panel neu-card">
        <div className="automode-header">
          <div>
            <h2 className="gradient-text">Autotrack</h2>
            <p className="muted">Select one prompt detector, then choose the cameras to monitor.</p>
          </div>
          <button type="button" className="link-btn automode-close" onClick={onClose}>Close</button>
        </div>

        {!modelAvailable && (
          <div className="demo-banner">{modelName} is unavailable. Check its runtime dependencies and model weights.</div>
        )}

        <label className="field-label" htmlFor="autotrack-model">Autotrack model</label>
        <select
          id="autotrack-model"
          className="automode-prompt"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="owlv2">OWLv2</option>
          <option value="yolo-world">YOLO-World</option>
        </select>

        <label className="automode-toggle-row">
          <span>Enable Autotrack</span>
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
        </label>

        <label className="field-label">Objects to detect (comma-separated)</label>
        <textarea
          className="automode-prompt"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="person, orange vest, hard hat"
        />
        <p className="muted" style={{ fontSize: '0.8rem', marginTop: 4 }}>
          Short labels work best, e.g. <code>person</code> or <code>person, blue jumpsuit</code>.
        </p>

        <label className="field-label">Cameras to watch</label>
        <div className="automode-cameras">
          {cameras.length === 0 && <p className="muted">No cameras found. Add cameras on the main page first.</p>}
          {cameras.map((c) => (
            <label key={c.camera_id} className={`cam-dropdown-item ${cameraIds.includes(c.camera_id) ? 'selected' : ''}`}>
              <input
                type="checkbox"
                checked={cameraIds.includes(c.camera_id)}
                onChange={() => toggleCamera(c.camera_id)}
              />
              {c.name || c.camera_id}
            </label>
          ))}
        </div>

        <ScheduleFields
          enabled={scheduleEnabled}
          start={scheduleStart}
          end={scheduleEnd}
          label="Watch only during"
          onChange={({ schedule_enabled, schedule_start, schedule_end }) => {
            setScheduleEnabled(schedule_enabled);
            setScheduleStart(schedule_start);
            setScheduleEnd(schedule_end);
          }}
        />

        <div className="conf-slider-wrap" style={{ marginTop: 12 }}>
          <span>Confidence</span>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
          />
          <span>{confidence.toFixed(2)}</span>
        </div>

        {activeCameras.length > 0 && (
          <p className="muted" style={{ fontSize: '0.82rem' }}>
            Watching: {activeCameras.join(', ')}
          </p>
        )}

        {status && <p className={`apply-status ${status.toLowerCase().includes('fail') || status.toLowerCase().includes('error') || status.startsWith('HTTP') ? 'err' : ''}`}>{status}</p>}

        <div className="automode-actions">
          <button type="button" className="btn" disabled={loading || (enabled && !modelAvailable)} onClick={handleApply}>
            {loading ? 'Applying…' : enabled ? 'Apply & Start Autotrack' : 'Save & Disable'}
          </button>
        </div>
      </div>
    </div>
  );
}
