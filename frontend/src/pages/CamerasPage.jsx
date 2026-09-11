import { useCallback, useEffect, useRef, useState } from 'react';
import CameraFeedTile from '../components/CameraFeedTile';
import { addCamera, listCameras, removeCamera, uploadVideo } from '../api/endpoints';

const GRIDS = [1, 2, 3];

let camerasCache = null;

export default function CamerasPage() {
  const [cameras, setCameras] = useState(() => camerasCache || []);
  const [grid, setGrid] = useState(2);
  const [name, setName] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(() => !camerasCache);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef(null);

  const load = useCallback(async () => {
    const showSpinner = !camerasCache;
    if (showSpinner) setLoading(true);
    try {
      const data = await listCameras();
      const list = Array.isArray(data) ? data : [];
      camerasCache = list;
      setCameras(list);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const onFileChosen = (e) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    setError('');
  };

  const onAddCamera = async () => {
    if (!selectedFile) {
      setError('Choose a video file first.');
      return;
    }
    setUploading(true);
    setError('');
    try {
      const up = await uploadVideo(selectedFile);
      const added = await addCamera({
        name: name.trim() || selectedFile.name,
        camera_id: cameraId.trim() || undefined,
        video_path: up.video_path,
      });
      if (added?.warning) setError(added.warning);
      setName('');
      setCameraId('');
      setSelectedFile(null);
      if (fileRef.current) fileRef.current.value = '';
      camerasCache = null;
      await load();
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const onRemove = async (cid) => {
    if (!window.confirm(`Remove camera ${cid}?`)) return;
    await removeCamera(cid);
    camerasCache = null;
    await load();
  };

  const badges = (cam) => {
    const b = [];
    if (cam.beta_active) b.push('Beta');
    if (!b.length) b.push('Live');
    return b;
  };

  return (
    <>
      <div className="page-header">
        <h1 className="page-title gradient-text">Camera Streams</h1>
        <p className="page-subtitle">Live demo video feeds with AI overlay support</p>
      </div>

      <div className="add-panel glass-panel neu-card">
        <div className="field">
          <label>Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Optional display name"
            disabled={uploading}
          />
        </div>
        <div className="field">
          <label>Camera ID</label>
          <input
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            placeholder="Auto-generated if empty"
            disabled={uploading}
          />
        </div>
        <div className="field field-grow">
          <label>Video file</label>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            onChange={onFileChosen}
            disabled={uploading}
          />
          {selectedFile && (
            <div className="file-picked muted">Selected: {selectedFile.name}</div>
          )}
        </div>
        <button
          type="button"
          className="btn-add-camera"
          onClick={onAddCamera}
          disabled={uploading || !selectedFile}
        >
          {uploading ? 'Adding…' : 'Add Camera'}
        </button>
        {error && <span className="err-inline">{error}</span>}
      </div>

      <div className="grid-controls glass-panel">
        <h2>{cameras.length} Camera{cameras.length !== 1 ? 's' : ''}</h2>
        <div className="grid-toggle">
          {GRIDS.map((n) => (
            <button key={n} type="button" className={grid === n ? 'active' : ''} onClick={() => setGrid(n)}>
              {n}×{n}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="page-loading">Loading cameras…</div>
      ) : cameras.length === 0 ? (
        <div className="empty-state glass-panel">No cameras — choose a video, then click Add Camera.</div>
      ) : (
        <div className={`camera-grid g${grid}`}>
          {cameras.map((cam, idx) => (
            <div key={cam.camera_id} className="cam-card glass-panel neu-card">
              <div className="cam-feed">
                <span className={`cam-dot ${cam.status === 'active' ? 'live' : 'off'}`} />
                <button type="button" className="cam-remove" onClick={() => onRemove(cam.camera_id)} title="Remove">×</button>
                <CameraFeedTile cameraId={cam.camera_id} index={idx} alt={cam.name || cam.camera_id} />
              </div>
              <div className="cam-info">
                <strong>{cam.name || cam.camera_id}</strong>
                <div className="badge-row">
                  {badges(cam).map((b) => (
                    <span key={b} className={`badge ${b === 'Beta' ? 'beta' : ''}`}>{b}</span>
                  ))}
                  <span className={`badge ${cam.status === 'active' ? 'ok' : 'off'}`}>{cam.status}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
