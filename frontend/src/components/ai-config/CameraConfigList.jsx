import { Link } from 'react-router-dom';
import { hasEntryExitZones, hasFlapgateZones, hasIntrusionZones, hasWorkforceZones } from './detectionTypes';

export default function CameraConfigList({
  cameras,
  configs,
  selectedClass,
  activeKey,
  checked,
  onToggle,
  zoneType,
  isActive,
}) {
  if (!cameras.length) {
    return (
      <div className="empty-state">
        <h3>No cameras found</h3>
        <p>
          <Link to="/">Add cameras first</Link>
        </p>
      </div>
    );
  }

  return (
    <div className="cam-list">
      {cameras.map((cam) => {
        const cid = cam.camera_id;
        const isOn = isActive ? isActive(cam) : Boolean(cam[activeKey]);
        const isLive = cam.status === 'active';
        const isChecked = checked.has(cid);
        const cfg = configs[cid] || {};

        let zoneBadge = null;
        let zoneLink = null;
        if (zoneType === 'entryexit') {
          const hasZones = hasEntryExitZones(cfg);
          zoneBadge = (
            <span className={`mini-badge ${hasZones ? 'mb-zone-ok' : 'mb-zone-no'}`}>
              {hasZones ? 'Zones Set' : 'No Zones'}
            </span>
          );
          zoneLink = (
            <Link to={`/zone-config/${encodeURIComponent(cid)}`} className="zone-link" onClick={(e) => e.stopPropagation()}>
              Config
            </Link>
          );
        } else if (zoneType === 'flapgate') {
          const hasZones = hasFlapgateZones(cfg);
          zoneBadge = (
            <span className={`mini-badge ${hasZones ? 'mb-zone-ok' : 'mb-zone-no'}`}>
              {hasZones ? '3 Zones Set' : 'No Zones'}
            </span>
          );
          zoneLink = (
            <Link
              to={`/flapgate-zone-config/${encodeURIComponent(cid)}`}
              className="zone-link"
              onClick={(e) => e.stopPropagation()}
            >
              Config
            </Link>
          );
        } else if (zoneType === 'workforce') {
          const hasZones = hasWorkforceZones(cfg);
          const awaySec = cfg.workforce?.config?.away_alert_sec;
          zoneBadge = (
            <>
              <span className={`mini-badge ${hasZones ? 'mb-zone-ok' : 'mb-zone-no'}`}>
                {hasZones ? 'Machine Zone Set' : 'No Zone'}
              </span>
              <span className="mini-badge mb-away">Away {awaySec != null ? awaySec : 3}s</span>
            </>
          );
          zoneLink = (
            <Link
              to={`/workforce-zone-config/${encodeURIComponent(cid)}`}
              className="zone-link"
              onClick={(e) => e.stopPropagation()}
            >
              Config
            </Link>
          );
        } else if (zoneType === 'intrusion') {
          const hasZones = hasIntrusionZones(cfg);
          zoneBadge = (
            <span className={`mini-badge ${hasZones ? 'mb-zone-ok' : 'mb-zone-no'}`}>
              {hasZones ? 'Zone + Prompt' : 'Not Configured'}
            </span>
          );
          zoneLink = (
            <Link
              to={`/intrusion-zone-config/${encodeURIComponent(cid)}`}
              className="zone-link"
              onClick={(e) => e.stopPropagation()}
            >
              Config
            </Link>
          );
        }

        return (
          <div
            key={cid}
            className={`cam-item ${isChecked ? selectedClass : ''}`}
            onClick={() => onToggle(cid)}
            onKeyDown={(e) => e.key === 'Enter' && onToggle(cid)}
            role="button"
            tabIndex={0}
          >
            <input
              type="checkbox"
              checked={isChecked}
              onChange={() => onToggle(cid)}
              onClick={(e) => e.stopPropagation()}
            />
            <div className={`cam-dot-inline ${isLive ? 'on' : 'off'}`} />
            <div className="cam-item-info">
              <div className="cam-item-name">{cam.name || cid}</div>
              <div className="cam-item-url">{cam.rtsp_url}</div>
            </div>
            <div className="cam-item-badges">
              <span className={`mini-badge ${isOn ? 'mb-on' : 'mb-off'}`}>{isOn ? 'Active' : 'Off'}</span>
              {zoneBadge}
              {zoneLink}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function useCameraSelection(cameras, activeKey, frActiveIds) {
  const initial = new Set();
  cameras.forEach((c) => {
    if (activeKey === 'fr') {
      if (frActiveIds?.includes(c.camera_id)) initial.add(c.camera_id);
    } else if (c[activeKey]) {
      initial.add(c.camera_id);
    }
  });
  return initial;
}
