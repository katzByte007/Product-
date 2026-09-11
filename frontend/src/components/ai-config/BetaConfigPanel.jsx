import { useEffect, useRef, useState } from 'react';
import { saveBetaConfig } from '../../api/endpoints';
import ScheduleFields from './ScheduleFields';

function betaPromptLabel(model) {
  return model === 'qwen' ? 'Prompt (instruction)' : 'Prompt (comma-separated labels)';
}

function betaPromptPlaceholder(model) {
  if (model === 'qwen') {
    return 'e.g. You are monitoring a food processing plant. Watch for missing hairnets, no gloves, unsanitized hands, and floor contamination. Raise alert only when violation is visible.';
  }
  return 'e.g. hairnet, gloves, lab coat, sanitizer, boots, hair cap';
}

function CameraDropdown({ cameras, selectedIds, onToggle }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  const selected = selectedIds || [];
  const label = selected.length
    ? `${selected.length} camera${selected.length === 1 ? '' : 's'} selected`
    : 'Select cameras…';

  useEffect(() => {
    const onDoc = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  return (
    <div className="cam-dropdown" ref={wrapRef}>
      <button type="button" className="cam-dropdown-btn" onClick={() => setOpen((v) => !v)}>
        {label}
      </button>
      {open && (
        <div className="cam-dropdown-list">
          {cameras.map((c) => {
            const id = c.camera_id;
            const sel = selected.includes(id);
            return (
              <label key={id} className={`cam-dropdown-item ${sel ? 'selected' : ''}`}>
                <input type="checkbox" checked={sel} onChange={() => onToggle(id)} />
                {c.name || id}
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function BetaConfigPanel({ cameras, betaMeta, setBetaMeta, onBack, onReload }) {
  const { owlv2Available, qwenAvailable, owlv2Message, qwenMessage, confidence, model, prompts, schedule_enabled, schedule_start, schedule_end } = betaMeta;
  const [status, setStatus] = useState({ msg: '', type: '' });
  const [applying, setApplying] = useState(false);

  const setModel = (m) => setBetaMeta((prev) => ({ ...prev, model: m }));
  const setConfidence = (c) => setBetaMeta((prev) => ({ ...prev, confidence: c }));

  const updatePrompt = (idx, text) => {
    setBetaMeta((prev) => {
      const next = [...prev.prompts];
      next[idx] = { ...next[idx], prompt_text: text };
      return { ...prev, prompts: next };
    });
  };

  const toggleCam = (idx, cameraId) => {
    setBetaMeta((prev) => {
      const next = prev.prompts.map((p, i) => {
        if (i !== idx) return p;
        const ids = [...(p.camera_ids || [])];
        const pos = ids.indexOf(cameraId);
        if (pos >= 0) ids.splice(pos, 1);
        else ids.push(cameraId);
        return { ...p, camera_ids: ids };
      });
      return { ...prev, prompts: next };
    });
  };

  const addPrompt = () => setBetaMeta((prev) => ({ ...prev, prompts: [...prev.prompts, { prompt_text: '', camera_ids: [] }] }));

  const removePrompt = (idx) => {
    setBetaMeta((prev) => {
      let next = prev.prompts.filter((_, i) => i !== idx);
      if (!next.length) next = [{ prompt_text: '', camera_ids: [] }];
      return { ...prev, prompts: next };
    });
  };

  const apply = async () => {
    if (model === 'owlv2' && !owlv2Available) {
      setStatus({ msg: 'Open vocabulary detection is not available. Install: pip install transformers pillow', type: 'err' });
      return;
    }
    if (model === 'qwen' && !qwenAvailable) {
      setStatus({ msg: qwenMessage || 'Instruction monitoring is not available.', type: 'err' });
      return;
    }
    const payload = {
      prompts: prompts.map((p) => ({ prompt_text: (p.prompt_text || '').trim(), camera_ids: p.camera_ids || [] })).filter((p) => p.prompt_text || p.camera_ids.length),
      confidence,
      model,
      schedule_enabled: Boolean(schedule_enabled),
      schedule_start: schedule_start || '08:00',
      schedule_end: schedule_end || '18:00',
    };
    setApplying(true);
    setStatus({ msg: 'Applying Beta configuration…', type: 'prog' });
    try {
      const data = await saveBetaConfig(payload);
      if (data.start_errors?.length) {
        setStatus({ msg: `Applied with errors: ${data.start_errors.map((e) => e.camera_id).join(', ')}`, type: 'err' });
      } else {
        const note = data.note ? ` ${data.note}` : '';
        setStatus({ msg: `Beta configuration applied — check Camera Streams for overlays.${note}`, type: 'ok' });
      }
      setTimeout(onReload, 800);
    } catch (e) {
      setStatus({ msg: e.message || 'Failed', type: 'err' });
    } finally {
      setApplying(false);
    }
  };

  const modelReady = model === 'owlv2' ? owlv2Available : qwenAvailable;
  const modelMessage = model === 'owlv2' ? owlv2Message : qwenMessage;

  return (
    <>
      <button type="button" className="back-btn" onClick={onBack}>← Back to AI Features</button>
      <div className="config-header">
        <h2><span className="tag-beta">Beta Model Detection</span> (Open Vocabulary / Instruction Monitoring)</h2>
      </div>
      <p className="muted" style={{ fontSize: '0.85em', marginBottom: 16 }}>
        Add one or more prompts.         With <strong>open vocabulary</strong>, each prompt is a comma-separated label list.
        With <strong>instruction monitoring</strong>, each prompt describes anomalies/safety events to monitor.
        For cleaner detections, use confidence <strong>0.35–0.50</strong> and specific labels (e.g. &quot;white hairnet, blue gloves&quot;).
      </p>
      <div className="ai-toolbar glass-panel">
        <div className="conf-slider-wrap">
          <label>Model:</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="owlv2">Open vocabulary</option>
            <option value="qwen">Instruction monitoring</option>
          </select>
        </div>
        <div className="conf-slider-wrap">
          <label>Confidence:</label>
          <input type="range" min={0.05} max={0.9} step={0.05} value={confidence} onChange={(e) => setConfidence(parseFloat(e.target.value))} />
          <span className="conf-val">{confidence}</span>
        </div>
        <ScheduleFields
          enabled={Boolean(schedule_enabled)}
          start={schedule_start || '08:00'}
          end={schedule_end || '18:00'}
          onChange={(next) => setBetaMeta((prev) => ({ ...prev, ...next }))}
        />
        <button type="button" className="btn-apply btn-apply-beta" disabled={applying} onClick={apply}>Apply Configuration</button>
      </div>

      {prompts.map((p, idx) => (
        <div key={idx} className="prompt-row glass-panel">
          <div className="prompt-input-wrap">
            <label>{betaPromptLabel(model)}</label>
            <input type="text" value={p.prompt_text} placeholder={betaPromptPlaceholder(model)} onChange={(e) => updatePrompt(idx, e.target.value)} />
          </div>
          <div className="prompt-cams">
            <label>Cameras for this prompt</label>
            <CameraDropdown cameras={cameras} selectedIds={p.camera_ids || []} onToggle={(id) => toggleCam(idx, id)} />
          </div>
          <button type="button" className="btn-remove-prompt" onClick={() => removePrompt(idx)}>Remove</button>
        </div>
      ))}

      <button type="button" className="add-prompt-btn" onClick={addPrompt}>+ Add another prompt</button>
      {status.msg && <div className={`apply-status ${status.type}`}>{status.msg}</div>}
      {modelMessage && (
        <div className={`beta-status ${modelReady ? 'ok' : 'err'}`}>{modelMessage}</div>
      )}
    </>
  );
}
