import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  cameraSnapshotUrl,
  enableFlapgate,
  getDetectionConfig,
} from '../api/endpoints';
import { CANVAS_H, CANVAS_W, useZoneCanvas } from '../hooks/useZoneCanvas';
import { pointsFromZoneArray } from '../utils/zoneCanvas';

const GATE_COLORS = {
  1: { stroke: '#f87171', fill: 'rgba(248,113,113,0.12)', label: 'Gate 1' },
  2: { stroke: '#34d399', fill: 'rgba(52,211,153,0.12)', label: 'Gate 2' },
  3: { stroke: '#60a5fa', fill: 'rgba(96,165,250,0.12)', label: 'Gate 3' },
};

export default function FlapgateZonePage() {
  const { cameraId } = useParams();
  const [snapshotUrl, setSnapshotUrl] = useState('');
  const [loadError, setLoadError] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const [gatePoints, setGatePoints] = useState({ 1: [], 2: [], 3: [] });
  const [currentGate, setCurrentGate] = useState(1);
  const [gateFinished, setGateFinished] = useState({ 1: false, 2: false, 3: false });
  const [confidence, setConfidence] = useState(0.65);
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
        const c = cfg?.flapgate?.config;
        if (!c) return;
        const gz = c.gate_zones || {};
        const next = { 1: [], 2: [], 3: [] };
        const fin = { 1: false, 2: false, 3: false };
        for (let gid = 1; gid <= 3; gid += 1) {
          const pts = gz[String(gid)] || [];
          if (pts.length >= 3) {
            next[gid] = pointsFromZoneArray(pts);
            fin[gid] = true;
          }
        }
        setGatePoints(next);
        setGateFinished(fin);
        if (c.confidence != null) setConfidence(c.confidence);
      })
      .catch(() => {});
  }, [cameraId]);

  const layers = useMemo(
    () =>
      [1, 2, 3].map((gid) => ({
        points: gatePoints[gid],
        stroke: GATE_COLORS[gid].stroke,
        fill: GATE_COLORS[gid].fill,
        closed: gateFinished[gid],
        label: GATE_COLORS[gid].label,
      })),
    [gatePoints, gateFinished]
  );

  const { canvasRef, imageReady, handleClick } = useZoneCanvas(snapshotUrl, layers);

  const setGatePts = (gid, updater) => {
    setGatePoints((prev) => ({
      ...prev,
      [gid]: typeof updater === 'function' ? updater(prev[gid]) : updater,
    }));
  };

  const finishZone = (ptsOverride, gateOverride) => {
    const gate = gateOverride ?? currentGate;
    const pts = ptsOverride ?? gatePoints[gate];
    if (pts.length < 3) {
      window.alert('Need at least 3 points');
      return;
    }
    setGateFinished((f) => ({ ...f, [gate]: true }));
    setGatePoints((prev) => {
      for (let gid = 1; gid <= 3; gid += 1) {
        if (gid !== gate && prev[gid].length < 3) {
          setCurrentGate(gid);
          break;
        }
      }
      return prev;
    });
  };

  const addPoint = (pt) => {
    if (gateFinished[currentGate]) {
      setGatePts(currentGate, []);
      setGateFinished((f) => ({ ...f, [currentGate]: false }));
    }
    setGatePts(currentGate, (pts) => {
      const next = [...pts, pt];
      if (next.length >= 12) {
        queueMicrotask(() => finishZone(next, currentGate));
      }
      return next;
    });
  };

  const undoPoint = () => {
    setGatePts(currentGate, (pts) => pts.slice(0, -1));
    setGateFinished((f) => ({ ...f, [currentGate]: false }));
  };

  const clearCurrent = () => {
    setGatePts(currentGate, []);
    setGateFinished((f) => ({ ...f, [currentGate]: false }));
  };

  const clearAll = () => {
    setGatePoints({ 1: [], 2: [], 3: [] });
    setGateFinished({ 1: false, 2: false, 3: false });
    setCurrentGate(1);
  };

  const saveAndStart = async () => {
    for (let gid = 1; gid <= 3; gid += 1) {
      if (gatePoints[gid].length < 3) {
        setStatus({ msg: `Gate ${gid} zone needs at least 3 points.`, ok: false });
        return;
      }
    }
    try {
      await enableFlapgate(cameraId, {
        gate_zones: {
          1: gatePoints[1],
          2: gatePoints[2],
          3: gatePoints[3],
        },
        canvas_width: CANVAS_W,
        canvas_height: CANVAS_H,
        confidence,
      });
      setStatus({ msg: 'Gate zones saved and trespassing detection started successfully.', ok: true });
    } catch (e) {
      setStatus({ msg: e.message || 'Failed to save', ok: false });
    }
  };

  return (
    <div className="zone-page">
      <Link to="/ai-config" className="back-btn">← Back to AI Features</Link>
      <h2>
        Flap Gate Zone Configuration — <span className="accent-fg">{cameraId}</span>
      </h2>

      <div className="instructions">
        <strong>How to draw zones:</strong>
        <br />
        1. Select <span className="c-g1">Gate 1</span>, <span className="c-g2">Gate 2</span>, or{' '}
        <span className="c-g3">Gate 3</span> below
        <br />
        2. Click on the camera snapshot to place polygon points (minimum 3 per gate)
        <br />
        3. Click <strong>Finish Zone</strong> when done, then draw the next gate zone
        <br />
        4. Once all 3 gate zones are drawn, click <strong>Save &amp; Start Detection</strong>
      </div>

      <div className="zone-controls">
        {[1, 2, 3].map((gid) => (
          <button
            key={gid}
            type="button"
            className={`zone-btn ${currentGate === gid ? `g${gid}-sel` : ''}`}
            onClick={() => setCurrentGate(gid)}
          >
            Gate {gid} Zone
          </button>
        ))}
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
        <span><span className="dot-g1" /> Gate 1: <strong>{gatePoints[1].length}</strong> pts</span>
        <span><span className="dot-g2" /> Gate 2: <strong>{gatePoints[2].length}</strong> pts</span>
        <span><span className="dot-g3" /> Gate 3: <strong>{gatePoints[3].length}</strong> pts</span>
        <span>Active: <strong>Gate {currentGate}</strong></span>
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
        <span className="conf-val-fg">{confidence}</span>
      </div>

      <button type="button" className="btn-zone-success-fg" onClick={saveAndStart}>
        Save &amp; Start Detection
      </button>

      {status.msg && (
        <div className={`zone-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</div>
      )}
    </div>
  );
}
