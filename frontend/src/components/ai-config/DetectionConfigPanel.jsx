import { useEffect, useRef, useState } from 'react';
import {
  disableEntryExit,
  disableFireSmoke,
  disableAnpr,
  disableFlapgate,
  disableHeadcount,
  disableIntrusion,
  disablePpe,
  disableWorkforce,
  enableEntryExit,
  enableFireSmoke,
  enableAnpr,
  enableFlapgate,
  enableHeadcount,
  enableIntrusion,
  enablePpe,
  enableWorkforce,
} from '../../api/endpoints';
import ApplyStatus from './ApplyStatus';
import CameraConfigList from './CameraConfigList';
import ScheduleFields from './ScheduleFields';
import { getDetectionMeta, hasEntryExitZones, hasFlapgateZones, hasIntrusionZones, hasWorkforceZones, savedAwayAlertSec, savedConfidence, savedSchedule, savedSpeedThreshold } from './detectionTypes';

const ENABLE_MAP = {
  headcount: enableHeadcount,
  entryexit: enableEntryExit,
  flapgate: enableFlapgate,
  workforce: enableWorkforce,
  intrusion: enableIntrusion,
  ppe: enablePpe,
  firesmoke: enableFireSmoke,
  anpr: enableAnpr,
};

const DISABLE_MAP = {
  headcount: disableHeadcount,
  entryexit: disableEntryExit,
  flapgate: disableFlapgate,
  workforce: disableWorkforce,
  intrusion: disableIntrusion,
  ppe: disablePpe,
  firesmoke: disableFireSmoke,
  anpr: disableAnpr,
};

export default function DetectionConfigPanel({ typeId, cameras, configs, onBack, onReload }) {
  const meta = getDetectionMeta(typeId);
  const [confidence, setConfidence] = useState(meta.defaultConf);
  const [speedThreshold, setSpeedThreshold] = useState(meta.defaultSpeedThreshold ?? 50);
  const [awayAlertSec, setAwayAlertSec] = useState(meta.defaultAwaySec ?? 3);
  const [scheduleEnabled, setScheduleEnabled] = useState(false);
  const [scheduleStart, setScheduleStart] = useState('08:00');
  const [scheduleEnd, setScheduleEnd] = useState('18:00');
  const [checked, setChecked] = useState(new Set());
  const [status, setStatus] = useState({ msg: '', type: '' });
  const [applying, setApplying] = useState(false);
  const dirtyRef = useRef(false);

  // Sync from server only when opening this panel or after configs finish loading — not on background polls.
  useEffect(() => {
    dirtyRef.current = false;
  }, [typeId]);

  useEffect(() => {
    if (dirtyRef.current) return;
    setConfidence(savedConfidence(cameras, configs, meta.configKey, meta.defaultConf));
    if (typeId === 'anpr') {
      setSpeedThreshold(
        savedSpeedThreshold(cameras, configs, meta.configKey, meta.defaultSpeedThreshold ?? 50)
      );
    }
    if (typeId === 'workforce') {
      setAwayAlertSec(
        savedAwayAlertSec(cameras, configs, meta.configKey, meta.defaultAwaySec ?? 3)
      );
    }
    const sched = savedSchedule(cameras, configs, meta.configKey);
    setScheduleEnabled(sched.enabled);
    setScheduleStart(sched.start);
    setScheduleEnd(sched.end);
    const sel = new Set();
    cameras.forEach((c) => {
      if (c[meta.activeKey]) sel.add(c.camera_id);
    });
    setChecked(sel);
  }, [typeId, cameras, configs, meta.activeKey, meta.configKey, meta.defaultConf, meta.defaultSpeedThreshold, meta.defaultAwaySec]);

  const toggleCam = (cid) => {
    dirtyRef.current = true;
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(cid)) next.delete(cid);
      else next.add(cid);
      return next;
    });
  };

  const selectAll = (on) => {
    dirtyRef.current = true;
    if (on) setChecked(new Set(cameras.map((c) => c.camera_id)));
    else setChecked(new Set());
  };

  const apply = async () => {
    const enableFn = ENABLE_MAP[typeId];
    const disableFn = DISABLE_MAP[typeId];
    setApplying(true);
    setStatus({ msg: `Applying ${meta.label.toLowerCase()} configuration...`, type: 'prog' });

    let ok = 0;
    let fail = 0;
    let noZone = 0;
    let lastErr = '';

    const scheduleBody = {
      schedule_enabled: scheduleEnabled,
      schedule_start: scheduleStart,
      schedule_end: scheduleEnd,
    };

    for (const cam of cameras) {
      const cid = cam.camera_id;
      const want = checked.has(cid);
      const has = cam[meta.activeKey];
      const cfg = configs[cid] || {};

      try {
        if (want) {
          if (typeId === 'entryexit') {
            const c = cfg.entryexit?.config || {};
            if (!hasEntryExitZones(cfg)) {
              noZone++;
              continue;
            }
            await enableFn(cid, {
              entry_zone: c.entry_zone,
              exit_zone: c.exit_zone,
              canvas_width: c.canvas_width || 800,
              canvas_height: c.canvas_height || 450,
              confidence,
              ...scheduleBody,
            });
          } else if (typeId === 'flapgate') {
            const c = cfg.flapgate?.config || {};
            const gz = c.gate_zones || {};
            if (!hasFlapgateZones(cfg)) {
              noZone++;
              continue;
            }
            await enableFn(cid, {
              gate_zones: gz,
              canvas_width: c.canvas_width || 800,
              canvas_height: c.canvas_height || 450,
              confidence,
              ...scheduleBody,
            });
          } else if (typeId === 'workforce') {
            const c = cfg.workforce?.config || {};
            if (!hasWorkforceZones(cfg)) {
              noZone++;
              continue;
            }
            await enableFn(cid, {
              machine_zone: c.machine_zone,
              canvas_width: c.canvas_width || 800,
              canvas_height: c.canvas_height || 450,
              confidence,
              away_alert_sec: awayAlertSec,
              ...scheduleBody,
            });
          } else if (typeId === 'intrusion') {
            const c = cfg.intrusion?.config || {};
            if (!hasIntrusionZones(cfg)) {
              noZone++;
              continue;
            }
            await enableFn(cid, {
              intrusion_zone: c.intrusion_zone,
              prompt: c.prompt,
              canvas_width: c.canvas_width || 800,
              canvas_height: c.canvas_height || 450,
              confidence,
              ...scheduleBody,
            });
          } else if (typeId === 'anpr') {
            await enableFn(cid, { confidence, speed_threshold_kmh: speedThreshold, ...scheduleBody });
          } else {
            await enableFn(cid, { confidence, ...scheduleBody });
          }
          ok++;
        } else if (!want && has) {
          await disableFn(cid);
          ok++;
        }
      } catch (e) {
        fail++;
        lastErr = e?.message || String(e);
      }
    }

    setApplying(false);
    let msg = `${ok} camera(s) updated.`;
    if (noZone > 0) {
      msg += ` ${noZone} camera(s) skipped (zones/prompt not configured — use Config link).`;
    }
    if (fail > 0) msg += ` ${fail} error(s)${lastErr ? `: ${lastErr}` : ''}.`;
    const type =
      fail > 0 || noZone > 0 ? 'err' : 'ok';
    if (fail === 0 && noZone === 0 && typeId !== 'entryexit' && typeId !== 'flapgate') {
      msg = `Configuration applied successfully. ${ok} camera(s) updated.`;
    }
    setStatus({ msg, type });
    setTimeout(onReload, 800);
  };

  return (
    <>
      <button type="button" className="back-btn" onClick={onBack}>
        ← Back to AI Features
      </button>
      <div className="config-header">
        <h2>
          <span className={meta.tagClass}>{meta.label}</span> Configuration
        </h2>
      </div>
      <div className="ai-toolbar">
        <div className="select-all-wrap">
          <input
            type="checkbox"
            id={`${typeId}SelectAll`}
            checked={cameras.length > 0 && checked.size === cameras.length}
            onChange={(e) => selectAll(e.target.checked)}
          />
          <label htmlFor={`${typeId}SelectAll`}>Select All</label>
        </div>
        <div className="conf-slider-wrap">
          <label>Confidence:</label>
          <input
            type="range"
            min={meta.confMin}
            max={0.9}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
          />
          <span className="conf-val">{confidence}</span>
        </div>
        {typeId === 'anpr' && (
          <div className="conf-slider-wrap">
            <label>Speed limit (km/h):</label>
            <input
              type="range"
              min={10}
              max={150}
              step={5}
              value={speedThreshold}
              onChange={(e) => setSpeedThreshold(parseFloat(e.target.value))}
            />
            <span className="conf-val">{speedThreshold}</span>
          </div>
        )}
        {typeId === 'workforce' && (
          <div className="conf-slider-wrap">
            <label>Away alert (sec):</label>
            <input
              type="range"
              min={1}
              max={60}
              step={1}
              value={awayAlertSec}
              onChange={(e) => {
                dirtyRef.current = true;
                setAwayAlertSec(parseFloat(e.target.value));
              }}
            />
            <span className="conf-val">{awayAlertSec}</span>
          </div>
        )}
        <ScheduleFields
          enabled={scheduleEnabled}
          start={scheduleStart}
          end={scheduleEnd}
          onChange={({ schedule_enabled, schedule_start, schedule_end }) => {
            dirtyRef.current = true;
            setScheduleEnabled(schedule_enabled);
            setScheduleStart(schedule_start);
            setScheduleEnd(schedule_end);
          }}
        />
        <button
          type="button"
          className={`btn-apply ${meta.applyClass}`}
          disabled={applying}
          onClick={apply}
        >
          Apply to Selected
        </button>
      </div>
      {typeId === 'workforce' && (
        <div className="away-config-banner glass-panel">
          <strong>Worker ID:1</strong> is the monitored operator. Away alert fires after <strong>{awayAlertSec}s</strong> off the machine.
          The machine zone turns <strong>red</strong> while they are away. Set the timer above, then Apply.
        </div>
      )}
      <CameraConfigList
        cameras={cameras}
        configs={configs}
        selectedClass={meta.selectedClass}
        activeKey={meta.activeKey}
        checked={checked}
        onToggle={toggleCam}
        zoneType={meta.zoneType}
      />
      <ApplyStatus message={status.msg} type={status.type} />
    </>
  );
}
