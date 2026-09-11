import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  cameraSnapshotUrl,
  enableEntryExit,
  getDetectionConfig,
} from '../api/endpoints';
import { CANVAS_H, CANVAS_W, useZoneCanvas } from '../hooks/useZoneCanvas';
import { pointsFromZoneArray } from '../utils/zoneCanvas';

export default function EntryExitZonePage() {
  const { cameraId } = useParams();
  const [snapshotUrl, setSnapshotUrl] = useState('');
  const [loadError, setLoadError] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const [entryPoints, setEntryPoints] = useState([]);
  const [exitPoints, setExitPoints] = useState([]);
  const [currentZone, setCurrentZone] = useState('entry');
  const [zoneFinished, setZoneFinished] = useState({ entry: false, exit: false });
  const [confidence, setConfidence] = useState(0.5);
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
        setSnapshotUrl('');
      }
    })();
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [cameraId, reloadKey]);

  useEffect(() => {
    getDetectionConfig(cameraId)
      .then((cfg) => {
        const c = cfg?.entryexit?.config;
        if (!c) return;
        if (c.entry_zone?.length >= 3) {
          setEntryPoints(pointsFromZoneArray(c.entry_zone));
          setZoneFinished((z) => ({ ...z, entry: true }));
        }
        if (c.exit_zone?.length >= 3) {
          setExitPoints(pointsFromZoneArray(c.exit_zone));
          setZoneFinished((z) => ({ ...z, exit: true }));
        }
        if (c.confidence != null) setConfidence(c.confidence);
      })
      .catch(() => {});
  }, [cameraId]);

  const layers = useMemo(
    () => [
      {
        points: entryPoints,
        stroke: '#fbbf24',
        fill: 'rgba(251,191,36,0.12)',
        closed: zoneFinished.entry,
        label: 'ENTRY',
      },
      {
        points: exitPoints,
        stroke: '#f87171',
        fill: 'rgba(248,113,113,0.12)',
        closed: zoneFinished.exit,
        label: 'EXIT',
      },
    ],
    [entryPoints, exitPoints, zoneFinished]
  );

  const { canvasRef, imageReady, handleClick } = useZoneCanvas(snapshotUrl, layers);

  const finishZone = (ptsOverride, zoneOverride) => {
    const zone = zoneOverride ?? currentZone;
    const pts = ptsOverride ?? (zone === 'entry' ? entryPoints : exitPoints);
    if (pts.length < 3) {
      window.alert('Need at least 3 points');
      return;
    }
    setZoneFinished((z) => ({ ...z, [zone]: true }));
    if (zone === 'entry') {
      setExitPoints((exit) => {
        if (exit.length < 3) setCurrentZone('exit');
        return exit;
      });
    } else {
      setEntryPoints((entry) => {
        if (entry.length < 3) setCurrentZone('entry');
        return entry;
      });
    }
  };

  const addPoint = (pt) => {
    if (zoneFinished[currentZone]) {
      if (currentZone === 'entry') setEntryPoints([]);
      else setExitPoints([]);
      setZoneFinished((z) => ({ ...z, [currentZone]: false }));
    }
    const setter = currentZone === 'entry' ? setEntryPoints : setExitPoints;
    setter((prev) => {
      const next = [...prev, pt];
      if (next.length >= 12) {
        queueMicrotask(() => finishZone(next, currentZone));
      }
      return next;
    });
  };

  const undoPoint = () => {
    const setter = currentZone === 'entry' ? setEntryPoints : setExitPoints;
    setter((pts) => pts.slice(0, -1));
    setZoneFinished((z) => ({ ...z, [currentZone]: false }));
  };

  const clearCurrent = () => {
    if (currentZone === 'entry') setEntryPoints([]);
    else setExitPoints([]);
    setZoneFinished((z) => ({ ...z, [currentZone]: false }));
  };

  const clearAll = () => {
    setEntryPoints([]);
    setExitPoints([]);
    setZoneFinished({ entry: false, exit: false });
    setCurrentZone('entry');
  };

  const saveAndStart = async () => {
    if (entryPoints.length < 3 || exitPoints.length < 3) {
      setStatus({ msg: 'Both zones need at least 3 points each.', ok: false });
      return;
    }
    try {
      await enableEntryExit(cameraId, {
        entry_zone: entryPoints,
        exit_zone: exitPoints,
        canvas_width: CANVAS_W,
        canvas_height: CANVAS_H,
        confidence,
      });
      setStatus({ msg: 'Zones saved and detection started successfully.', ok: true });
    } catch (e) {
      setStatus({ msg: e.message || 'Failed to save', ok: false });
    }
  };

  return (
    <div className="zone-page">
      <Link to="/ai-config" className="back-btn">← Back to AI Features</Link>
      <h2>
        Entry / Exit Zone Configuration — <span className="accent-ee">{cameraId}</span>
      </h2>

      <div className="instructions">
        <strong>How to draw zones:</strong>
        <br />
        1. Select <span className="c-entry">ENTRY Zone</span> or <span className="c-exit">EXIT Zone</span> below
        <br />
        2. Click on the camera snapshot to place polygon points (minimum 3)
        <br />
        3. Click <strong>Finish Zone</strong> when done, then draw the other zone
        <br />
        4. Click <strong>Save &amp; Start Detection</strong> to apply
      </div>

      <div className="zone-controls">
        <button
          type="button"
          className={`zone-btn ${currentZone === 'entry' ? 'entry-sel' : ''}`}
          onClick={() => setCurrentZone('entry')}
        >
          ENTRY Zone
        </button>
        <button
          type="button"
          className={`zone-btn ${currentZone === 'exit' ? 'exit-sel' : ''}`}
          onClick={() => setCurrentZone('exit')}
        >
          EXIT Zone
        </button>
      </div>

      <div className="canvas-wrapper">
        <div className="canvas-overlay">Click to add points</div>
        {(!imageReady || loadError) && (
          <div className="snapshot-loading">{loadError || 'Loading camera snapshot...'}</div>
        )}
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{ display: imageReady && !loadError ? 'block' : 'none' }}
          onClick={(e) => {
            const pt = handleClick(e);
            if (pt) addPoint(pt);
          }}
        />
      </div>

      <div className="point-info">
        <span><span className="dot-entry" /> Entry: <strong>{entryPoints.length}</strong> points</span>
        <span><span className="dot-exit" /> Exit: <strong>{exitPoints.length}</strong> points</span>
        <span>Active: <strong>{currentZone}</strong></span>
      </div>

      <div className="action-bar">
        <button type="button" className="btn-zone-ghost" onClick={undoPoint}>Undo</button>
        <button type="button" className="btn-zone-ghost" onClick={() => finishZone()}>Finish Zone</button>
        <button type="button" className="btn-zone-ghost" onClick={clearCurrent}>Clear Current</button>
        <button type="button" className="btn-zone-danger" onClick={clearAll}>Clear All</button>
        <button type="button" className="btn-zone-ghost" onClick={() => setReloadKey((k) => k + 1)}>Refresh Snapshot</button>
      </div>

      <div className="conf-row">
        <label>Detection Confidence:</label>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.05}
          value={confidence}
          onChange={(e) => setConfidence(parseFloat(e.target.value))}
        />
        <span className="conf-val-ee">{confidence}</span>
      </div>

      <button type="button" className="btn-zone-success-ee" onClick={saveAndStart}>
        Save &amp; Start Detection
      </button>

      {status.msg && (
        <div className={`zone-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</div>
      )}
    </div>
  );
}
