import api from './client';

export const login = (username, password) => api.post('/api/login', { username, password });
export const logout = () => api.post('/api/logout', {});
export const authMe = () => api.get('/api/auth/me');

export const listCameras = () => api.get('/api/cameras');
export const addCamera = (payload) => api.post('/api/cameras', payload);
export const removeCamera = (cameraId) => api.delete(`/api/cameras/${encodeURIComponent(cameraId)}`);
export const uploadVideo = async (file) => {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/cameras/upload', { method: 'POST', body: fd, credentials: 'include' });
  let data = {};
  try {
    data = await res.json();
  } catch {
    throw new Error(res.ok ? 'Upload failed' : `Upload failed (HTTP ${res.status})`);
  }
  if (!res.ok) throw new Error(data.error || `Upload failed (HTTP ${res.status})`);
  return data;
};
export const cameraSnapshotUrl = (cameraId) => `/api/cameras/${encodeURIComponent(cameraId)}/snapshot`;
export const feedSnapshotUrl = (cameraId) => `/api/feed_snapshot/${encodeURIComponent(cameraId)}`;
export const liveFeedUrl = (cameraId) => `/video_feed/live/${encodeURIComponent(cameraId)}`;

export const getDetectionConfig = (cameraId) => api.get(`/api/detection/config/${encodeURIComponent(cameraId)}`);
export const listAllDetectionConfigs = () => api.get('/api/detection/configs/all');
export const getBetaConfig = () => api.get('/api/detection/beta/config');
export const saveBetaConfig = (payload) => api.post('/api/detection/beta/save', payload);

export const enableWorkforce = (cameraId, body) => api.post(`/api/detection/workforce/enable/${encodeURIComponent(cameraId)}`, body);
export const disableWorkforce = (cameraId) => api.post(`/api/detection/workforce/disable/${encodeURIComponent(cameraId)}`, {});
export const enableIntrusion = (cameraId, body) => api.post(`/api/detection/intrusion/enable/${encodeURIComponent(cameraId)}`, body);
export const disableIntrusion = (cameraId) => api.post(`/api/detection/intrusion/disable/${encodeURIComponent(cameraId)}`, {});
export const saveDetectionConfig = (cameraId, detectionType, body) =>
  api.put(`/api/detection/config/${encodeURIComponent(cameraId)}/${encodeURIComponent(detectionType)}`, body);

export const enableHeadcount = (cameraId, body) => api.post(`/api/detection/headcount/enable/${encodeURIComponent(cameraId)}`, body);
export const disableHeadcount = (cameraId) => api.post(`/api/detection/headcount/disable/${encodeURIComponent(cameraId)}`, {});
export const enableEntryExit = (cameraId, body) => api.post(`/api/detection/entryexit/enable/${encodeURIComponent(cameraId)}`, body);
export const disableEntryExit = (cameraId) => api.post(`/api/detection/entryexit/disable/${encodeURIComponent(cameraId)}`, {});
export const enableFlapgate = (cameraId, body) => api.post(`/api/detection/flapgate/enable/${encodeURIComponent(cameraId)}`, body);
export const disableFlapgate = (cameraId) => api.post(`/api/detection/flapgate/disable/${encodeURIComponent(cameraId)}`, {});
export const enablePpe = (cameraId, body) => api.post(`/api/detection/ppe/enable/${encodeURIComponent(cameraId)}`, body);
export const disablePpe = (cameraId) => api.post(`/api/detection/ppe/disable/${encodeURIComponent(cameraId)}`, {});
export const enableFireSmoke = (cameraId, body) => api.post(`/api/detection/firesmoke/enable/${encodeURIComponent(cameraId)}`, body);
export const disableFireSmoke = (cameraId) => api.post(`/api/detection/firesmoke/disable/${encodeURIComponent(cameraId)}`, {});
export const enableAnpr = (cameraId, body) => api.post(`/api/detection/anpr/enable/${encodeURIComponent(cameraId)}`, body);
export const disableAnpr = (cameraId) => api.post(`/api/detection/anpr/disable/${encodeURIComponent(cameraId)}`, {});

export const frStatus = () => api.get('/api/fr/status');
export const frDatasets = () => api.get('/api/fr/datasets');
export const frScanSync = () => api.post('/api/fr/scan_and_sync', {});
export const frFolderAdd = (payload) => api.post('/api/fr/folder/add', payload);
export const frFolderDelete = (payload) => api.post('/api/fr/folder/delete', payload);
export const frCollectStart = (payload) => api.post('/api/fr/collect/start', payload);
export const frCollectStop = (payload) => api.post('/api/fr/collect/stop', payload);
export const frCollectStatus = (sessionId) => api.get(`/api/fr/collect/status/${encodeURIComponent(sessionId)}`);
export const frEnable = (cameraId, body) => api.post(`/api/fr/detection/enable/${encodeURIComponent(cameraId)}`, body);
export const frDisable = (cameraId) => api.post(`/api/fr/detection/disable/${encodeURIComponent(cameraId)}`, {});

export const listAlerts = (detectionType = '') => {
  const q = detectionType ? `?detection_type=${encodeURIComponent(detectionType)}` : '';
  return api.get(`/api/alerts${q}`);
};
export const deleteAlert = (alertId) => api.delete(`/api/alerts/${alertId}`);
export const deleteAllAlerts = () => api.delete('/api/alerts');
export const assignAlert = (alertId, engineerId) =>
  api.post(`/api/alerts/${alertId}/assign`, { engineer_id: engineerId });
export const getAutoModeConfig = () => api.get('/api/automode/config');
export const saveAutoModeConfig = (payload) => api.post('/api/automode/config', payload);
export const listResponseEngineers = () => api.get('/api/system/response-engineers');
export const addResponseEngineer = (payload) => api.post('/api/system/response-engineers', payload);
export const deleteResponseEngineer = (id) => api.delete(`/api/system/response-engineers/${id}`);
export const getSmtpSettings = () => api.get('/api/system/smtp');
export const saveSmtpSettings = (payload) => api.post('/api/system/smtp', payload);
export const testSmtpSettings = (payload) => api.post('/api/system/smtp/test', payload);
export const listAssignedEngineers = () => api.get('/api/assigned-engineers');
export const getPlantMap = () => api.get('/api/plant-map');
export const liveStats = () => api.get('/api/stats');
export const systemStats = () => api.get('/api/system-stats');
export const health = () => api.get('/api/health');

export const listUsers = () => api.get('/api/system/users');
export const onlineUsers = () => api.get('/api/system/online-users');
export const getRecordSchedules = () => api.get('/api/system/record-schedules');
export const getHolidays = () => api.get('/api/system/holidays');
export const getLocalSettings = () => api.get('/api/system/local-settings');

export const enableDetection = (type, cameraId, body = {}) =>
  api.post(`/api/detection/${type}/enable/${encodeURIComponent(cameraId)}`, body);
export const disableDetection = (type, cameraId) =>
  api.post(`/api/detection/${type}/disable/${encodeURIComponent(cameraId)}`, {});
