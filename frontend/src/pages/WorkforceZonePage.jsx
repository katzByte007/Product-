import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { cameraSnapshotUrl, getDetectionConfig, saveDetectionConfig } from '../api/endpoints';
import { CANVAS_H, CANVAS_W, useZoneCanvas } from '../hooks/useZoneCanvas';
import { pointsFromZoneArray } from '../utils/zoneCanvas';

export default function WorkforceZonePage() {
  const { cameraId } = useParams();
  const [snapshotUrl, setSnapshotUrl] = useState('');
  const [loadError, setLoadError] = useState('');
  const [points, setPoints] = useState([]);
  const [zoneClosed, setZoneClosed] = useState(false);
  const [confidence, setConfidence] = useState(0.25);
  const [awayAlertSec, setAwayAlertSec] = useState(3);
  const [status, setStatus] = useState({ msg: '', ok: null });

  useEffect(() => {
    let objectUrl = null;
    (async () => {
      setLoadError('');
      try {
        const res = await fetch(cameraSnapshotUrl(cameraId), { credentials: 'include' });
        if (!res.ok) throw new Error('No snapshot');
        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        setSnapshotUrl(objectUrl);
      } catch {
        setLoadError('Could not load snapshot. Make sure the camera is connected.');
      }
    })();
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [cameraId]);

  useEffect(() => {
    getDetectionConfig(cameraId).then((cfg) => {
      const c = cfg?.workforce?.config;
      if (!c) return;
      if (c.machine_zone?.length >= 3) {
        setPoints(pointsFromZoneArray(c.machine_zone));
        setZoneClosed(true);
      }
      if (c.confidence != null) setConfidence(c.confidence);
      if (c.away_alert_sec != null) setAwayAlertSec(Number(c.away_alert_sec));
    }).catch(() => {});
  }, [cameraId]);

  const layers = useMemo(
    () => [{ points, stroke: '#38bdf8', fill: 'rgba(56,189,248,0.15)', closed: zoneClosed, label: 'MACHINE' }],
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

  const finishZone = () => {
    if (points.length < 3) {
      window.alert('Need at least 3 points for the machine zone');
      return;
    }
    setZoneClosed(true);
  };

  const saveZones = async () => {
    if (points.length < 3) {
      setStatus({ msg: 'Draw the machine zone (minimum 3 points).', ok: false });
      return;
    }
    try {
      await saveDetectionConfig(cameraId, 'workforce', {
        machine_zone: points,
        canvas_width: CANVAS_W,
        canvas_height: CANVAS_H,
        confidence,
        away_alert_sec: awayAlertSec,
      });
      setStatus({ msg: 'Machine zone saved. Go back to AI Features → Workforce Monitoring → Apply.', ok: true });
    } catch (e) {
      setStatus({ msg: e.message || 'Failed to save', ok: false });
    }
  };

  return (
    <div className="zone-page">
      <Link to="/ai-config" className="back-btn">← Back to AI Features</Link>
      <h2>Workforce Monitoring — Machine Zone — <span className="accent-ee">{cameraId}</span></h2>
      <div className="instructions">
        Draw the <strong>machine zone</strong> where the worker should stay. Click points on the snapshot, then Finish Zone and Save.
        Set how many seconds away from the machine should raise an alert. After saving, enable Workforce Monitoring on the AI Features page.
      </div>
      <div className="zone-controls">
        <button type="button" className="btn-zone" onClick={() => { setPoints([]); setZoneClosed(false); }}>Clear</button>
        <button type="button" className="btn-zone" onClick={() => setPoints((p) => p.slice(0, -1))}>Undo</button>
        <button type="button" className="btn-zone primary" onClick={finishZone}>Finish Zone</button>
        <label className="conf-inline">Confidence: {confidence}</label>
        <input type="range" min={0.1} max={0.9} step={0.05} value={confidence} onChange={(e) => setConfidence(parseFloat(e.target.value))} />
        <label className="conf-inline">Away alert: {awayAlertSec}s</label>
        <input type="range" min={1} max={60} step={1} value={awayAlertSec} onChange={(e) => setAwayAlertSec(parseFloat(e.target.value))} />
        <button type="button" className="btn-save-zones" onClick={saveZones}>Save Zone</button>
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} onClick={onCanvasClick} />
        {loadError && <div className="snapshot-loading">{loadError}</div>}
      </div>
      {status.msg && <div className={`apply-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</div>}
    </div>
  );
}
