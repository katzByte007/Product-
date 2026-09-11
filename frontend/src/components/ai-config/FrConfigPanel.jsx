import { useCallback, useEffect, useRef, useState } from 'react';
import {
  frCollectStart,
  frCollectStatus,
  frCollectStop,
  frDatasets,
  frDisable,
  frEnable,
  frFolderAdd,
  frFolderDelete,
  frScanSync,
  frStatus,
  getDetectionConfig,
  listCameras,
} from '../../api/endpoints';
import ApplyStatus from './ApplyStatus';
import CameraConfigList from './CameraConfigList';
import ScheduleFields from './ScheduleFields';

export default function FrConfigPanel({ cameras, frActiveCameras, onBack, onReload, setFrActiveCameras }) {
  const [tab, setTab] = useState(0);
  const [frAvailable, setFrAvailable] = useState({ insightface: false, yolo: false, deepsort: false });
  const [datasets, setDatasets] = useState([]);
  const [baseDir, setBaseDir] = useState('');
  const [dataStatus, setDataStatus] = useState({ msg: '', type: '' });
  const [syncMsg, setSyncMsg] = useState('');
  const [syncing, setSyncing] = useState(false);
  const [showAddFolder, setShowAddFolder] = useState(false);
  const [newPersonName, setNewPersonName] = useState('');

  const [collectCam, setCollectCam] = useState('');
  const [collectRtsp, setCollectRtsp] = useState('');
  const [collectPerson, setCollectPerson] = useState('');
  const [collectSessionId, setCollectSessionId] = useState(null);
  const [collectStatus, setCollectStatus] = useState({ msg: '', type: '' });
  const [collectCount, setCollectCount] = useState(0);
  const [collectCompleted, setCollectCompleted] = useState([]);
  const [collectRunning, setCollectRunning] = useState(false);
  const pollRef = useRef(null);
  const syncInFlightRef = useRef(false);

  const [threshold, setThreshold] = useState(0.3);
  const [drawThreshold, setDrawThreshold] = useState(0.55);
  const [detThreshold, setDetThreshold] = useState(0.25);
  const [checked, setChecked] = useState(new Set());
  const [frApplyStatus, setFrApplyStatus] = useState({ msg: '', type: '' });
  const [applying, setApplying] = useState(false);
  const [scheduleEnabled, setScheduleEnabled] = useState(false);
  const [scheduleStart, setScheduleStart] = useState('08:00');
  const [scheduleEnd, setScheduleEnd] = useState('18:00');

  const loadFrData = useCallback(async () => {
    try {
      const [statusData, datasetsData, camList] = await Promise.all([
        frStatus(),
        frDatasets(),
        listCameras(),
      ]);
      setFrAvailable({
        insightface: statusData.insightface_available,
        yolo: statusData.yolo_available,
        deepsort: statusData.deepsort_available,
      });
      setFrActiveCameras(statusData.active_cameras || []);
      setDatasets(datasetsData.datasets || []);
      setBaseDir(datasetsData.base_dir || '');
      const active = statusData.active_cameras || [];
      setChecked(new Set(active));
      const defaults = statusData.defaults || {};
      setThreshold(Number(defaults.threshold ?? 0.3));
      setDrawThreshold(Number(defaults.draw_threshold ?? 0.55));
      setDetThreshold(Number(defaults.det_threshold ?? 0.25));
      if (active.length > 0) {
        try {
          const cfg = await getDetectionConfig(active[0]);
          const frCfg = cfg?.fr_detection?.config;
          if (frCfg) {
            if (frCfg.threshold != null) setThreshold(Number(frCfg.threshold));
            if (frCfg.draw_threshold != null) setDrawThreshold(Number(frCfg.draw_threshold));
            if (frCfg.det_threshold != null) setDetThreshold(Number(frCfg.det_threshold));
            if (frCfg.schedule_enabled != null) setScheduleEnabled(Boolean(frCfg.schedule_enabled));
            if (frCfg.schedule_start) setScheduleStart(frCfg.schedule_start);
            if (frCfg.schedule_end) setScheduleEnd(frCfg.schedule_end);
          }
        } catch {
          /* keep defaults */
        }
      }
    } catch (e) {
      console.error('FR load error:', e);
    }
  }, [setFrActiveCameras]);

  const syncFolders = useCallback(
    async (silent = false) => {
      if (syncInFlightRef.current) return;
      syncInFlightRef.current = true;
      setSyncing(true);
      if (!silent) setSyncMsg('Scanning folders…');
      try {
        const data = await frScanSync();
        if (data.success) {
          if (!data.skipped) {
            setSyncMsg(data.message || `Synced ${data.total_synced} folder(s).`);
          }
        } else {
          setSyncMsg(`Sync error: ${data.error || 'unknown'}`);
        }
      } catch (e) {
        setSyncMsg(`Sync failed: ${e.message}`);
      } finally {
        syncInFlightRef.current = false;
        setSyncing(false);
        await loadFrData();
      }
    },
    [loadFrData]
  );

  useEffect(() => {
    syncFolders(true);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [syncFolders]);

  const resetCollectUi = () => {
    setCollectSessionId(null);
    setCollectRunning(false);
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const onCollectCamChange = (camId) => {
    setCollectCam(camId);
    const cam = cameras.find((c) => c.camera_id === camId);
    if (cam) setCollectRtsp(cam.rtsp_url || '');
  };

  const startCollect = async () => {
    if (!collectRtsp.trim()) {
      setCollectStatus({ msg: 'Enter an RTSP URL.', type: 'err' });
      return;
    }
    if (!collectPerson.trim()) {
      setCollectStatus({ msg: 'Enter a person name.', type: 'err' });
      return;
    }
    if (!frAvailable.yolo) {
      setCollectStatus({ msg: 'YOLO not available.', type: 'err' });
      return;
    }
    if (!frAvailable.deepsort) {
      setCollectStatus({ msg: 'DeepSort not available.', type: 'err' });
      return;
    }

    setCollectStatus({ msg: 'Starting collection...', type: 'prog' });
    try {
      const data = await frCollectStart({
        rtsp_url: collectRtsp.trim(),
        person_name: collectPerson.trim(),
      });
      setCollectSessionId(data.session_id);
      setCollectRunning(true);
      setCollectCount(0);
      setCollectCompleted([]);
      setCollectStatus({ msg: 'Collection running. Stand in front of the camera.', type: 'prog' });

      pollRef.current = setInterval(async () => {
        try {
          const st = await frCollectStatus(data.session_id);
          if (st.error) {
            setCollectStatus({ msg: `Error: ${st.error}`, type: 'err' });
            resetCollectUi();
            return;
          }
          const counts = st.counts || {};
          const maxImages = Math.max(0, ...Object.values(counts));
          setCollectCount(maxImages);
          if (st.completed?.length) setCollectCompleted(st.completed);
          if (st.stopped) {
            resetCollectUi();
            setCollectStatus({ msg: 'Collection stopped.', type: 'ok' });
            loadFrData();
          }
        } catch (e) {
          console.error('FR poll error:', e);
        }
      }, 1500);
    } catch (e) {
      setCollectStatus({ msg: e.message || 'Failed to start.', type: 'err' });
    }
  };

  const stopCollect = async () => {
    if (!collectSessionId) return;
    try {
      await frCollectStop({ session_id: collectSessionId });
    } catch {
      /* ignore */
    }
    resetCollectUi();
    setCollectStatus({ msg: 'Stopping...', type: 'prog' });
    setTimeout(loadFrData, 1500);
  };

  const addFolder = async () => {
    const name = newPersonName.trim();
    if (!name) {
      setDataStatus({ msg: 'Person name is required.', type: 'err' });
      return;
    }
    try {
      await frFolderAdd({ person_name: name });
      setNewPersonName('');
      setShowAddFolder(false);
      setDataStatus({ msg: `Folder created for "${name}".`, type: 'ok' });
      loadFrData();
    } catch (e) {
      setDataStatus({ msg: e.message || 'Failed to add folder.', type: 'err' });
    }
  };

  const deleteFolder = async (personName) => {
    if (!window.confirm(`Delete all face data for "${personName}"? This cannot be undone.`)) return;
    try {
      await frFolderDelete({ person_name: personName });
      setDataStatus({ msg: `Deleted data for "${personName}".`, type: 'ok' });
      loadFrData();
    } catch (e) {
      setDataStatus({ msg: e.message || 'Delete failed.', type: 'err' });
    }
  };

  const toggleCam = (cid) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(cid)) next.delete(cid);
      else next.add(cid);
      return next;
    });
  };

  const applyFr = async () => {
    if (!frAvailable.insightface) {
      setFrApplyStatus({
        msg: 'InsightFace is not available. Install: pip install insightface onnxruntime',
        type: 'err',
      });
      return;
    }
    if (checked.size === 0 && frActiveCameras.length === 0) {
      setFrApplyStatus({ msg: 'Select at least one camera to enable FR.', type: 'err' });
      return;
    }

    setApplying(true);
    setFrApplyStatus({ msg: 'Applying FR configuration...', type: 'prog' });
    let ok = 0;
    let fail = 0;
    const failMsgs = [];

    for (const cam of cameras) {
      const cid = cam.camera_id;
      const want = checked.has(cid);
      const has = frActiveCameras.includes(cid);
      try {
        if (want) {
          await frEnable(cid, {
            threshold,
            draw_threshold: drawThreshold,
            det_threshold: detThreshold,
            schedule_enabled: scheduleEnabled,
            schedule_start: scheduleStart,
            schedule_end: scheduleEnd,
          });
          ok++;
        } else if (!want && has) {
          await frDisable(cid);
          ok++;
        }
      } catch (e) {
        fail++;
        failMsgs.push(`${cid}: ${e.message}`);
      }
    }

    setApplying(false);
    await loadFrData();
    onReload();
    if (fail === 0) {
      setFrApplyStatus({ msg: `FR configuration applied. ${ok} camera(s) updated.`, type: 'ok' });
    } else {
      const detail = failMsgs.length ? ` First error: ${failMsgs[0]}` : '';
      setFrApplyStatus({
        msg: `Completed with ${fail} error(s). ${ok} camera(s) updated.${detail}`,
        type: 'err',
      });
    }
  };

  const handleBack = () => {
    resetCollectUi();
    onBack();
  };

  return (
    <>
      <button type="button" className="back-btn" onClick={handleBack}>
        ← Back to AI Features
      </button>
      <div className="config-header">
        <h2>
          <span className="tag-fr">FR Detection</span> — Face Recognition
        </h2>
      </div>

      {!frAvailable.insightface && (
        <div className="fr-unavail-banner">
          InsightFace is not available. Install: <code>pip install insightface onnxruntime</code>
        </div>
      )}
      {!frAvailable.yolo && (
        <div className="fr-unavail-banner">
          YOLO (ultralytics) not available. Install: <code>pip install ultralytics</code>
        </div>
      )}
      {!frAvailable.deepsort && (
        <div className="fr-unavail-banner">
          DeepSort not available. Install: <code>pip install deep-sort-realtime</code>
        </div>
      )}

      <div className="fr-tabs">
        {['Face Data', 'Collect Data', 'Live Detection'].map((label, i) => (
          <button
            key={label}
            type="button"
            className={`fr-tab ${tab === i ? 'active' : ''}`}
            onClick={() => setTab(i)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 0 && (
        <>
          <p className="muted" style={{ fontSize: '0.82em', marginBottom: 16 }}>
            Manage employee face image folders. Each row corresponds to a person folder under{' '}
            <code style={{ color: '#60a5fa' }}>{baseDir}</code>. Use <em>Collect Data</em> to populate a folder with 50
            captured images, or place images manually and click <strong>Sync Folders → CSV</strong>.
          </p>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', marginBottom: 18 }}>
            <button type="button" className="btn-fr-add" onClick={() => setShowAddFolder(true)}>
              + Add Person Folder
            </button>
            <button
              type="button"
              className="btn-fr-add"
              style={{ borderStyle: 'solid', background: 'rgba(96,165,250,0.07)', color: '#60a5fa' }}
              disabled={syncing}
              onClick={() => syncFolders(false)}
            >
              {syncing ? '⟳ Syncing...' : '↻ Sync Folders → CSV'}
            </button>
            {syncMsg && <span className="muted" style={{ fontSize: '0.8em' }}>{syncMsg}</span>}
          </div>
          {showAddFolder && (
            <div className="card" style={{ padding: 18, marginBottom: 18 }}>
              <label>Person Name</label>
              <input
                type="text"
                value={newPersonName}
                placeholder="e.g. John_Doe"
                onChange={(e) => setNewPersonName(e.target.value)}
                style={{ display: 'block', width: 320, margin: '6px 0 10px' }}
              />
              <button type="button" className="btn-apply btn-apply-fr" onClick={addFolder} style={{ marginRight: 8 }}>
                Create Folder
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowAddFolder(false)}>
                Cancel
              </button>
            </div>
          )}
          <table className="fr-dataset-table">
            <thead>
              <tr>
                <th>Person Name</th>
                <th>Images Captured</th>
                <th>Folder Path</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {!datasets.length ? (
                <tr>
                  <td colSpan={4} style={{ color: '#475569', textAlign: 'center', padding: 32 }}>
                    No face datasets yet. Add a person folder or use Collect Data.
                  </td>
                </tr>
              ) : (
                datasets.map((d) => (
                  <tr key={d.person_name}>
                    <td style={{ fontWeight: 600 }}>{d.person_name}</td>
                    <td>
                      <span style={{ color: d.image_count >= 50 ? '#34d399' : '#fbbf24' }}>{d.image_count}</span>
                      {d.image_count < 50 ? (
                        <span style={{ color: '#64748b', fontSize: '0.75em' }}> / 50 needed</span>
                      ) : (
                        <span style={{ color: '#34d399', fontSize: '0.75em' }}> ✓</span>
                      )}
                    </td>
                    <td style={{ color: '#475569', fontSize: '0.8em', wordBreak: 'break-all' }}>{d.folder}</td>
                    <td>
                      <button type="button" className="btn-fr-del" onClick={() => deleteFolder(d.person_name)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <ApplyStatus message={dataStatus.msg} type={dataStatus.type} />
        </>
      )}

      {tab === 1 && (
        <>
          <p className="muted" style={{ fontSize: '0.82em', marginBottom: 16 }}>
            Select a camera, enter the employee name, and click <strong>Start Collection</strong>. The system captures 50
            face images and updates the CSV when complete.
          </p>
          <div className="fr-collect-layout">
            <div className="fr-collect-panel">
              <label>Camera (RTSP URL auto-filled)</label>
              <select value={collectCam} onChange={(e) => onCollectCamChange(e.target.value)}>
                <option value="">— Select camera —</option>
                {cameras.map((c) => (
                  <option key={c.camera_id} value={c.camera_id}>
                    {c.name || c.camera_id}
                  </option>
                ))}
              </select>
              <label>RTSP URL</label>
              <input
                type="text"
                value={collectRtsp}
                placeholder="rtsp://..."
                onChange={(e) => setCollectRtsp(e.target.value)}
              />
              <label>Person Name (folder will be created if not exists)</label>
              <input
                type="text"
                value={collectPerson}
                placeholder="e.g. John_Doe"
                onChange={(e) => setCollectPerson(e.target.value)}
              />
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                {!collectRunning && (
                  <button type="button" className="btn-apply btn-apply-fr" onClick={startCollect}>
                    ▶ Start Collection
                  </button>
                )}
                {collectRunning && (
                  <button type="button" className="btn btn-ghost" onClick={stopCollect}>
                    ■ Stop
                  </button>
                )}
              </div>
              <ApplyStatus message={collectStatus.msg} type={collectStatus.type} />
              {collectRunning && (
                <div style={{ marginTop: 14 }}>
                  <div className="muted" style={{ fontSize: '0.82em', marginBottom: 4 }}>
                    Collecting for: <strong>{collectPerson}</strong> | Images:{' '}
                    <strong>{collectCount}</strong> / 50
                  </div>
                  <div className="fr-progress-bar">
                    <div
                      className="fr-progress-bar-inner"
                      style={{ width: `${Math.min(100, (collectCount / 50) * 100)}%` }}
                    />
                  </div>
                  {collectCompleted.length > 0 && (
                    <div style={{ fontSize: '0.78em', color: '#34d399', marginTop: 4 }}>
                      ✓ Completed IDs: {collectCompleted.join(', ')} — CSV updated
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="fr-collect-preview-wrap">
              <p className="fr-preview-title">Live camera preview</p>
              <p className="muted fr-preview-hint">
                {collectCam
                  ? collectRunning
                    ? `Face boxes show who is being tracked. Confirm only "${collectPerson}" is in frame before images are saved.`
                    : 'Select the correct person in view before starting collection — other faces in the room may otherwise be captured.'
                  : 'Select a camera to preview the stream.'}
              </p>
              {collectCam ? (
                <img
                  src={
                    collectSessionId
                      ? `/video_feed/fr_collect/${encodeURIComponent(collectSessionId)}`
                      : `/video_feed/raw/${encodeURIComponent(collectCam)}`
                  }
                  alt="Camera preview for face collection"
                  className="fr-live-preview"
                />
              ) : (
                <div className="fr-preview-placeholder">No camera selected</div>
              )}
              {collectPerson.trim() && collectCam && (
                <div className="fr-preview-person-tag">
                  Target person: <strong>{collectPerson.trim()}</strong>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {tab === 2 && (
        <>
          <p className="muted" style={{ fontSize: '0.82em', marginBottom: 16 }}>
            Select cameras to run live face recognition on. Matches against collected face embeddings.
          </p>
          <div className="ai-toolbar">
            <div className="select-all-wrap">
              <input
                type="checkbox"
                id="frSelectAll"
                checked={cameras.length > 0 && checked.size === cameras.length}
                onChange={(e) => {
                  if (e.target.checked) setChecked(new Set(cameras.map((c) => c.camera_id)));
                  else setChecked(new Set());
                }}
              />
              <label htmlFor="frSelectAll">Select All</label>
            </div>
            <div className="fr-threshold-inputs">
              <label className="fr-threshold-field">
                Recognition threshold
                <input
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value) || 0.3)}
                />
              </label>
              <label className="fr-threshold-field">
                Draw threshold
                <input
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={drawThreshold}
                  onChange={(e) => setDrawThreshold(parseFloat(e.target.value) || 0.55)}
                />
              </label>
              <label className="fr-threshold-field">
                Detection threshold
                <input
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={detThreshold}
                  onChange={(e) => setDetThreshold(parseFloat(e.target.value) || 0.25)}
                />
              </label>
            </div>
            <p className="muted fr-threshold-help">
              Recognition = name match score. Draw = minimum score to show a label. Detection = face
              finder sensitivity (lower helps distant faces).
            </p>
            <ScheduleFields
              enabled={scheduleEnabled}
              start={scheduleStart}
              end={scheduleEnd}
              onChange={({ schedule_enabled, schedule_start, schedule_end }) => {
                setScheduleEnabled(schedule_enabled);
                setScheduleStart(schedule_start);
                setScheduleEnd(schedule_end);
              }}
            />
            <button type="button" className="btn-apply btn-apply-fr" disabled={applying} onClick={applyFr}>
              Apply to Selected
            </button>
          </div>
          <CameraConfigList
            cameras={cameras}
            configs={{}}
            selectedClass="selected-fr"
            checked={checked}
            onToggle={toggleCam}
            isActive={(cam) => frActiveCameras.includes(cam.camera_id)}
          />
          <ApplyStatus message={frApplyStatus.msg} type={frApplyStatus.type} />
        </>
      )}
    </>
  );
}
