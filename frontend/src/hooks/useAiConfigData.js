import { useCallback, useEffect, useState } from 'react';
import {
  frStatus,
  getBetaConfig,
  getDetectionConfig,
  listAllDetectionConfigs,
  listCameras,
} from '../api/endpoints';

const CONFIG_BATCH = 6;

async function loadConfigsBatched(cameraIds) {
  const cfgMap = {};
  for (let i = 0; i < cameraIds.length; i += CONFIG_BATCH) {
    const batch = cameraIds.slice(i, i + CONFIG_BATCH);
    const results = await Promise.all(
      batch.map((id) =>
        getDetectionConfig(id)
          .then((cfg) => ({ id, cfg }))
          .catch(() => ({ id, cfg: {} }))
      )
    );
    results.forEach((r) => {
      cfgMap[r.id] = r.cfg;
    });
  }
  return cfgMap;
}

function countActive(cameras, key, configs, configKey, enabledRows, dtype) {
  const runtime = cameras.filter((c) => c[key]).length;
  const fromCfg = Object.values(configs).filter((c) => c?.[configKey]?.enabled).length;
  const fromAll = (enabledRows || []).filter((r) => r.detection_type === dtype).length;
  return Math.max(runtime, fromCfg, fromAll);
}

function computeStats(cameras, configs, enabledRows, frActiveCameras) {
  return {
    headcount: countActive(cameras, 'headcount_active', configs, 'headcount', enabledRows, 'headcount'),
    entryexit: countActive(cameras, 'entryexit_active', configs, 'entryexit', enabledRows, 'entryexit'),
    flapgate: countActive(cameras, 'flapgate_active', configs, 'flapgate', enabledRows, 'flapgate'),
    workforce: countActive(cameras, 'workforce_active', configs, 'workforce', enabledRows, 'workforce'),
    intrusion: countActive(cameras, 'intrusion_active', configs, 'intrusion', enabledRows, 'intrusion'),
    ppe: countActive(cameras, 'ppe_active', configs, 'ppe', enabledRows, 'ppe'),
    firesmoke: countActive(cameras, 'fire_smoke_active', configs, 'firesmoke', enabledRows, 'firesmoke'),
    anpr: countActive(cameras, 'anpr_active', configs, 'anpr', enabledRows, 'anpr'),
    beta: cameras.filter((c) => c.beta_active).length,
    fr: Math.max(
      frActiveCameras.length,
      cameras.filter((c) => c.fr_active).length,
      enabledRows.filter((r) => r.detection_type === 'fr_detection').length
    ),
    total: cameras.length,
  };
}

export function useAiConfigData(view = 'types') {
  const [cameras, setCameras] = useState([]);
  const [configs, setConfigs] = useState({});
  const [enabledRows, setEnabledRows] = useState([]);
  const [betaMeta, setBetaMeta] = useState({
    owlv2Available: false,
    qwenAvailable: false,
    owlv2Message: '',
    qwenMessage: '',
    confidence: 0.35,
    model: 'owlv2',
    prompts: [{ prompt_text: '', camera_ids: [] }],
    schedule_enabled: false,
    schedule_start: '08:00',
    schedule_end: '18:00',
  });
  const [frActiveCameras, setFrActiveCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [configsLoading, setConfigsLoading] = useState(false);
  const [stats, setStats] = useState({
    headcount: 0,
    entryexit: 0,
    flapgate: 0,
    workforce: 0,
    intrusion: 0,
    ppe: 0,
    firesmoke: 0,
    anpr: 0,
    beta: 0,
    fr: 0,
    total: 0,
  });

  const loadData = useCallback(async ({ blocking = true, withConfigs = false } = {}) => {
    if (blocking) setLoading(true);
    try {
      const [camList, betaData, frData, allConfigs] = await Promise.all([
        listCameras(),
        getBetaConfig().catch(() => ({})),
        frStatus().catch(() => ({ active_cameras: [] })),
        listAllDetectionConfigs().catch(() => []),
      ]);

      const cams = Array.isArray(camList) ? camList : [];
      const rows = Array.isArray(allConfigs) ? allConfigs : [];
      const frActive = frData.active_cameras || [];

      setCameras(cams);
      setFrActiveCameras(frActive);
      setEnabledRows(rows);

      const prompts = (betaData.prompts || []).map((p) => ({
        prompt_text: p.prompt_text || '',
        camera_ids: p.camera_ids || [],
      }));
      setBetaMeta({
        owlv2Available: betaData.owlv2_available === true,
        qwenAvailable: betaData.qwen_available === true,
        owlv2Message: betaData.owlv2_message || '',
        qwenMessage: betaData.qwen_message || '',
        confidence: betaData.confidence != null ? betaData.confidence : 0.35,
        model: (betaData.beta_model || 'owlv2').toLowerCase() === 'qwen' ? 'qwen' : 'owlv2',
        prompts: prompts.length ? prompts : [{ prompt_text: '', camera_ids: [] }],
        schedule_enabled: Boolean(betaData.schedule_enabled),
        schedule_start: betaData.schedule_start || '08:00',
        schedule_end: betaData.schedule_end || '18:00',
      });

      if (withConfigs && cams.length) {
        setConfigsLoading(true);
        const cfgMap = await loadConfigsBatched(cams.map((c) => c.camera_id));
        setConfigs(cfgMap);
        setStats(computeStats(cams, cfgMap, rows, frActive));
      } else {
        setStats(computeStats(cams, {}, rows, frActive));
      }
    } catch (e) {
      console.error('AI config load error:', e);
    } finally {
      setLoading(false);
      setConfigsLoading(false);
    }
  }, []);

  const loadCameraConfigs = useCallback(async () => {
    if (!cameras.length) return;
    setConfigsLoading(true);
    try {
      const cfgMap = await loadConfigsBatched(cameras.map((c) => c.camera_id));
      setConfigs(cfgMap);
      setStats(computeStats(cameras, cfgMap, enabledRows, frActiveCameras));
    } catch (e) {
      console.error('AI camera config load error:', e);
    } finally {
      setConfigsLoading(false);
    }
  }, [cameras, enabledRows, frActiveCameras]);

  const refreshStats = useCallback(async () => {
    try {
      const [camList, allConfigs, frData] = await Promise.all([
        listCameras(),
        listAllDetectionConfigs().catch(() => []),
        frStatus().catch(() => ({ active_cameras: [] })),
      ]);
      const cams = Array.isArray(camList) ? camList : [];
      const rows = Array.isArray(allConfigs) ? allConfigs : [];
      const frActive = frData.active_cameras || [];
      setStats(computeStats(cams, configs, rows, frActive));
    } catch (e) {
      console.error('AI stats refresh error:', e);
    }
  }, [configs]);

  useEffect(() => {
    loadData({ blocking: true, withConfigs: false });
  }, [loadData]);

  useEffect(() => {
    if (view === 'types') return undefined;
    loadCameraConfigs();
  }, [view, loadCameraConfigs]);

  useEffect(() => {
    if (view !== 'types') return undefined;
    const id = setInterval(refreshStats, 5000);
    return () => clearInterval(id);
  }, [view, refreshStats]);

  return {
    cameras,
    configs,
    betaMeta,
    setBetaMeta,
    frActiveCameras,
    setFrActiveCameras,
    loading,
    configsLoading,
    reload: (opts = {}) => loadData({ blocking: false, withConfigs: view !== 'types', ...opts }),
    stats,
  };
}
