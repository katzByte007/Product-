export default function ScheduleFields({ enabled, start, end, onChange, label = 'Active hours' }) {
  return (
    <div className="schedule-wrap">
      <label className="schedule-toggle">
        <input
          type="checkbox"
          checked={Boolean(enabled)}
          onChange={(e) => onChange({ schedule_enabled: e.target.checked, schedule_start: start, schedule_end: end })}
        />
        {label}
      </label>
      <input
        type="time"
        value={start || '08:00'}
        disabled={!enabled}
        onChange={(e) => onChange({ schedule_enabled: enabled, schedule_start: e.target.value, schedule_end: end })}
      />
      <span className="schedule-to">to</span>
      <input
        type="time"
        value={end || '18:00'}
        disabled={!enabled}
        onChange={(e) => onChange({ schedule_enabled: enabled, schedule_start: start, schedule_end: e.target.value })}
      />
      <span className="schedule-hint">{enabled ? 'Runs only in this window' : 'Always on'}</span>
    </div>
  );
}

export function scheduleFromConfig(cfg, fallback = { enabled: false, start: '08:00', end: '18:00' }) {
  if (!cfg) return fallback;
  return {
    enabled: Boolean(cfg.schedule_enabled),
    start: cfg.schedule_start || fallback.start,
    end: cfg.schedule_end || fallback.end,
  };
}
