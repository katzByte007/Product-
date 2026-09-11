import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { cameraSnapshotUrl, getDetectionConfig, saveDetectionConfig } from '../api/endpoints';
import { CANVAS_H, CANVAS_W, useZoneCanvas } from '../hooks/useZoneCanvas';
import { pointsFromZoneArray } from '../utils/zoneCanvas';

export default function IntrusionZonePage() {
  const { cameraId } = useParams();
  const [snapshotUrl, setSnapshotUrl] = useState('');
  const [loadError, setLoadError] = useState('');
  const [points, setPoints] = useState([]);
  const [zoneClosed, setZoneClosed] = useState(false);
  const [prompt, setPrompt] = useState('person, intruder');
  const [confidence, setConfidence] = useState(0.25);
  const [status, setStatus] = useState({ msg: '', ok: null });

  useEffect(() => {
    let objectUrl = null;
    (async () => {
      try {
        const res = await fetch(cameraSnapshotUrl(cameraId), { credentials: 'include' });
        if (!res.ok) throw new Error('No snapshot');
        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        setSnapshotUrl(objectUrl);
      } catch {
        setLoadError('Could not load snapshot.');
      }
    })();
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [cameraId]);

  useEffect(() => {
    getDetectionConfig(cameraId).then((cfg) => {
      const c = cfg?.intrusion?.config;
      if (!c) return;
      if (c.intrusion_zone?.length >= 3) {
        setPoints(pointsFromZoneArray(c.intrusion_zone));
        setZoneClosed(true);
      }
      if (c.prompt) setPrompt(c.prompt);
      if (c.confidence != null) setConfidence(c.confidence);
    }).catch(() => {});
  }, [cameraId]);

  const layers = useMemo(
    () => [{ points, stroke: '#f97316', fill: 'rgba(249,115,22,0.15)', closed: zoneClosed, label: 'RESTRICTED' }],
    [points, zoneClosed]
  );

  const { canvasRef, handleClick } = useZoneCanvas(snapshotUrl, layers);

  const onCanvasClick = (e) => {
    const pt = handleClick(e);
    if (!pt) return;
    if (zoneClosed) {
      setPoints([]);
      setZoneClosed(false);
    }
    setPoints((prev) => [...prev, pt]);
  };

  const saveConfig = async () => {
    if (points.length < 3) {
      setStatus({ msg: 'Draw the intrusion zone (minimum 3 points).', ok: false });
      return;
    }
    if (!prompt.trim()) {
      setStatus({ msg: 'Enter OWLv2 labels (comma-separated), e.g. person, truck', ok: false });
      return;
    }
    try {
      await saveDetectionConfig(cameraId, 'intrusion', {
        intrusion_zone: points,
        prompt: prompt.trim(),
        canvas_width: CANVAS_W,
        canvas_height: CANVAS_H,
        confidence,
      });
      setStatus({ msg: 'Saved. Go to AI Features → Intrusion Detection → Apply on this camera.', ok: true });
    } catch (e) {
      setStatus({ msg: e.message || 'Failed to save', ok: false });
    }
  };

  return (
    <div className="zone-page">
      <Link to="/ai-config" className="back-btn">← Back to AI Features</Link>
      <h2>Intrusion Zone — <span className="accent-ee">{cameraId}</span></h2>
      <div className="instructions">
        Draw a restricted zone. Set open-vocabulary labels (OWLv2). People without an orange safety vest and orange safety helmet raise a No PPE alert. A person inside the zone raises an Intrusion alert. Both appear on Alerts with a person snapshot.
      </div>
      <div className="zone-controls stack">
        <label>Open vocabulary prompt (comma-separated)</label>
        <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="person, truck, forklift" />
        <label>Confidence: {confidence}</label>
        <input type="range" min={0.05} max={0.9} step={0.05} value={confidence} onChange={(e) => setConfidence(parseFloat(e.target.value))} />
      </div>
      <div className="zone-controls">
        <button type="button" className="btn-zone" onClick={() => { setPoints([]); setZoneClosed(false); }}>Clear</button>
        <button type="button" className="btn-zone" onClick={() => setPoints((p) => p.slice(0, -1))}>Undo</button>
        <button type="button" className="btn-zone primary" onClick={() => points.length >= 3 && setZoneClosed(true)}>Finish Zone</button>
        <button type="button" className="btn-save-zones" onClick={saveConfig}>Save Config</button>
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} onClick={onCanvasClick} />
        {loadError && <div className="snapshot-loading">{loadError}</div>}
      </div>
      {status.msg && <div className={`apply-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</div>}
    </div>
  );
}
