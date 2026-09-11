/** Detection categories for client filtering */
export const DETECTION_CATEGORIES = [
  { id: 'all', label: 'All Features' },
  { id: 'counting', label: 'Counting & Occupancy' },
  { id: 'access', label: 'Access & Zones' },
  { id: 'safety', label: 'Safety & Compliance' },
  { id: 'vehicle', label: 'Vehicle & ANPR' },
  { id: 'vision_ai', label: 'Vision AI / VLM' },
  { id: 'identity', label: 'Identity' },
];

/** Camera requirement tiers */
export const CAMERA_TIERS = [
  { id: 'all', label: 'All Requirements' },
  { id: 'standard', label: 'Standard — works with current cameras' },
  { id: 'high_resolution', label: 'High resolution (1080p+)' },
  { id: 'dedicated', label: 'Dedicated / special camera' },
];

/** Detection type metadata */
export const DETECTION_TYPES = [
  {
    id: 'headcount',
    category: 'counting',
    cameraTier: 'standard',
    cameraTierLabel: 'Standard camera',
    minResolution: '720p',
    dedicatedNote: '',
    cardClass: 'tc-hc',
    icon: 'HC',
    title: 'Head Count Detection',
    description:
      'Detects and counts heads in the frame using a custom-trained model. Best suited for overhead or elevated camera angles. Tracks unique IDs with Kalman filtering.',
    tagClass: 'tag-hc',
    label: 'Head Count',
    selectedClass: 'selected-hc',
    applyClass: 'btn-apply-hc',
    defaultConf: 0.1,
    confMin: 0.05,
    activeKey: 'headcount_active',
    configKey: 'headcount',
  },
  {
    id: 'entryexit',
    category: 'access',
    cameraTier: 'standard',
    cameraTierLabel: 'Standard + zone config',
    minResolution: '720p',
    dedicatedNote: '',
    cardClass: 'tc-ee',
    icon: 'E/E',
    title: 'Entry / Exit Zone Detection',
    description:
      'Tracks people crossing defined entry and exit zones to accurately count occupancy. Requires per-camera zone configuration. Uses person detection with zone-based state tracking.',
    tagClass: 'tag-ee',
    label: 'Entry / Exit',
    selectedClass: 'selected-ee',
    applyClass: 'btn-apply-ee',
    defaultConf: 0.5,
    confMin: 0.1,
    activeKey: 'entryexit_active',
    configKey: 'entryexit',
    zoneType: 'entryexit',
  },
  {
    id: 'flapgate',
    category: 'access',
    cameraTier: 'high_resolution',
    cameraTierLabel: '1080p+ recommended',
    minResolution: '1080p',
    dedicatedNote: 'Fixed view of flap gates; stable mounting required',
    cardClass: 'tc-fg',
    icon: 'FG',
    title: 'Flap Gate Trespassing',
    description:
      'Monitors 3 flap gates for trespassing. Detects gate color (green/red), tracks persons via segmentation, and alerts when multiple people enter during a single gate opening.',
    tagClass: 'tag-fg',
    label: 'Flap Gate Trespassing',
    selectedClass: 'selected-fg',
    applyClass: 'btn-apply-fg',
    defaultConf: 0.65,
    confMin: 0.1,
    activeKey: 'flapgate_active',
    configKey: 'flapgate',
    zoneType: 'flapgate',
  },
  {
    id: 'workforce',
    category: 'counting',
    cameraTier: 'standard',
    cameraTierLabel: 'Standard + machine zone',
    minResolution: '720p',
    dedicatedNote: '',
    cardClass: 'tc-wf',
    icon: 'WF',
    title: 'Workforce Monitoring',
    description:
      'Draw a machine zone per camera. Tracks the worker at the machine, accumulates dwell time, and raises an Away alert (with snapshot) after the time you set.',
    tagClass: 'tag-wf',
    label: 'Workforce Monitoring',
    selectedClass: 'selected-wf',
    applyClass: 'btn-apply-wf',
    defaultConf: 0.25,
    confMin: 0.1,
    defaultAwaySec: 3,
    activeKey: 'workforce_active',
    configKey: 'workforce',
    zoneType: 'workforce',
  },
  {
    id: 'intrusion',
    category: 'access',
    cameraTier: 'standard',
    cameraTierLabel: 'Standard + zone + OWLv2',
    minResolution: '720p',
    dedicatedNote: '',
    cardClass: 'tc-intr',
    icon: 'IZ',
    title: 'Intrusion Detection',
    description:
      'Draw a restricted zone and set OWLv2 labels. People and prompt objects are drawn on the live camera. A person without orange safety vest and helmet raises a No PPE alert; a person in the zone raises an Intrusion alert — both with snapshots.',
    tagClass: 'tag-intr',
    label: 'Intrusion Detection',
    selectedClass: 'selected-intr',
    applyClass: 'btn-apply-intr',
    defaultConf: 0.25,
    confMin: 0.05,
    activeKey: 'intrusion_active',
    configKey: 'intrusion',
    zoneType: 'intrusion',
  },
  {
    id: 'ppe',
    category: 'safety',
    cameraTier: 'high_resolution',
    cameraTierLabel: '1080p+ for small PPE items',
    minResolution: '1080p',
    dedicatedNote: '',
    cardClass: 'tc-ppe',
    icon: 'PPE',
    title: 'PPE Detection',
    description:
      'Detects Personal Protective Equipment compliance — helmets, vests, gloves, masks and more. Flags violations in real time using a custom-trained multi-class PPE model.',
    tagClass: 'tag-ppe',
    label: 'PPE Detection',
    selectedClass: 'selected-ppe',
    applyClass: 'btn-apply-ppe',
    defaultConf: 0.4,
    confMin: 0.1,
    activeKey: 'ppe_active',
    configKey: 'ppe',
  },
  {
    id: 'firesmoke',
    category: 'safety',
    cameraTier: 'standard',
    cameraTierLabel: 'Standard wide-area view',
    minResolution: '720p',
    dedicatedNote: '',
    cardClass: 'tc-fs',
    icon: 'F/S',
    title: 'Fire and Smoke Detection',
    description:
      'Detects fire and smoke in live camera streams using a dedicated custom model. Raises high-severity incident alerts with saved snapshot evidence.',
    tagClass: 'tag-fs',
    label: 'Fire and Smoke',
    selectedClass: 'selected-fs',
    applyClass: 'btn-apply-fs',
    defaultConf: 0.35,
    confMin: 0.1,
    activeKey: 'fire_smoke_active',
    configKey: 'firesmoke',
  },
  {
    id: 'anpr',
    category: 'vehicle',
    cameraTier: 'dedicated',
    cameraTierLabel: 'Dedicated LPR camera',
    minResolution: '1080p',
    dedicatedNote: 'Road-facing plate camera with IR; not suitable for general CCTV',
    cardClass: 'tc-anpr',
    icon: 'ANPR',
    title: 'ANPR Detection',
    description:
      'Automatic Number Plate Recognition using LibreYOLO. Detects vehicle type, reads license plates via OCR, estimates speed, and alerts on overspeeding.',
    tagClass: 'tag-anpr',
    label: 'ANPR Detection',
    selectedClass: 'selected-anpr',
    applyClass: 'btn-apply-anpr',
    defaultConf: 0.25,
    confMin: 0.1,
    defaultSpeedThreshold: 50,
    activeKey: 'anpr_active',
    configKey: 'anpr',
  },
  {
    id: 'beta',
    category: 'vision_ai',
    cameraTier: 'standard',
    cameraTierLabel: 'Standard (CPU/GPU for VLM)',
    minResolution: '720p',
    dedicatedNote: '',
    cardClass: 'tc-beta',
    icon: 'B',
    title: 'Beta Model Detection',
    description:
      'Open-vocabulary or instruction monitoring. Use comma-separated labels for open vocabulary; use monitoring instructions to raise anomaly alerts on live streams.',
    tagClass: 'tag-beta',
    label: 'Beta Model Detection',
    activeKey: 'beta_active',
  },
  {
    id: 'fr',
    category: 'identity',
    cameraTier: 'dedicated',
    cameraTierLabel: 'Dedicated face-level camera',
    minResolution: '1080p',
    dedicatedNote: 'Close frontal face view; turnstile or entry lane mounting',
    cardClass: 'tc-fr',
    icon: 'FR',
    title: 'FR Detection',
    description:
      'Face Recognition using InsightFace AntelopeV2. Collect face images per employee, manage datasets, and run live recognition on selected cameras with high-accuracy identity matching.',
    tagClass: 'tag-fr',
    label: 'FR Detection',
    selectedClass: 'selected-fr',
    applyClass: 'btn-apply-fr',
  },
];

export function getDetectionMeta(id) {
  return DETECTION_TYPES.find((t) => t.id === id);
}

/** True if detection can run on generic demo/standard cameras (not dedicated-only). */
export function isCompatibleWithStandardCameras(type) {
  return type.cameraTier === 'standard' || type.cameraTier === 'high_resolution';
}

export function filterDetectionTypes({ category = 'all', cameraTier = 'all', standardCamerasOnly = false } = {}) {
  return DETECTION_TYPES.filter((t) => {
    if (category !== 'all' && t.category !== category) return false;
    if (cameraTier !== 'all' && t.cameraTier !== cameraTier) return false;
    if (standardCamerasOnly && !isCompatibleWithStandardCameras(t)) return false;
    return true;
  });
}

export function savedConfidence(cameras, configs, configKey, fallback) {
  for (const cam of cameras) {
    const v = configs[cam.camera_id]?.[configKey]?.config?.confidence;
    if (v != null) return v;
  }
  return fallback;
}

export function savedSpeedThreshold(cameras, configs, configKey, fallback) {
  for (const cam of cameras) {
    const v = configs[cam.camera_id]?.[configKey]?.config?.speed_threshold_kmh;
    if (v != null) return v;
  }
  return fallback;
}

export function savedAwayAlertSec(cameras, configs, configKey, fallback) {
  for (const cam of cameras) {
    const v = configs[cam.camera_id]?.[configKey]?.config?.away_alert_sec;
    if (v != null) return v;
  }
  return fallback;
}

export function savedSchedule(cameras, configs, configKey) {
  for (const cam of cameras) {
    const c = configs[cam.camera_id]?.[configKey]?.config;
    if (c && (c.schedule_enabled || c.schedule_start || c.schedule_end)) {
      return {
        enabled: Boolean(c.schedule_enabled),
        start: c.schedule_start || '08:00',
        end: c.schedule_end || '18:00',
      };
    }
  }
  return { enabled: false, start: '08:00', end: '18:00' };
}

export function hasEntryExitZones(cfg) {
  const c = cfg?.entryexit?.config || {};
  return c.entry_zone && c.entry_zone.length >= 3;
}

export function hasFlapgateZones(cfg) {
  const gz = cfg?.flapgate?.config?.gate_zones || {};
  return gz['1']?.length >= 3 && gz['2']?.length >= 3 && gz['3']?.length >= 3;
}

export function hasWorkforceZones(cfg) {
  const z = cfg?.workforce?.config?.machine_zone || [];
  return z.length >= 3;
}

export function hasIntrusionZones(cfg) {
  const c = cfg?.intrusion?.config || {};
  return (c.intrusion_zone?.length >= 3) && Boolean((c.prompt || '').trim());
}
