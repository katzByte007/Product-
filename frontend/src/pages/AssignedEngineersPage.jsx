import { useEffect, useState } from 'react';
import { listAssignedEngineers, listResponseEngineers } from '../api/endpoints';

export default function AssignedEngineersPage() {
  const [assignments, setAssignments] = useState([]);
  const [engineers, setEngineers] = useState([]);

  useEffect(() => {
    const load = () => {
      Promise.all([listAssignedEngineers(), listResponseEngineers()])
        .then(([a, e]) => {
          setAssignments(a.assignments || []);
          setEngineers(e.engineers || []);
        })
        .catch(console.error);
    };
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <div className="page-header">
        <h1 className="page-title gradient-text">Assigned Engineers</h1>
        <p className="page-subtitle">Alerts assigned from the Alerts page, with email send status</p>
      </div>
      <div className="two-col">
        <section className="glass-panel neu-card">
          <h3>Assigned tasks</h3>
          {assignments.length === 0 ? (
            <p className="muted">No assignments yet. Open Alerts, pick an engineer, and click Assign &amp; email.</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Engineer</th>
                  <th>Department</th>
                  <th>Camera</th>
                  <th>Type</th>
                  <th>Alert</th>
                  <th>Email</th>
                </tr>
              </thead>
              <tbody>
                {assignments.map((a) => (
                  <tr key={a.id}>
                    <td>{a.engineer}</td>
                    <td>{a.department || '—'}</td>
                    <td>{a.camera_id}</td>
                    <td>{a.detection_type}</td>
                    <td className="cell-clip" title={a.message}>{a.message}</td>
                    <td className="cell-clip" title={a.email_error || a.status}>{a.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <section className="glass-panel neu-card">
          <h3>Response team</h3>
          {engineers.length === 0 ? (
            <p className="muted">Add engineers in System → Local (name, department, email).</p>
          ) : (
            <ul className="engineer-list">
              {engineers.map((e) => (
                <li key={e.id} className="engineer-item">
                  <strong>{e.name}</strong>
                  <span className="muted">{e.department || 'No department'}</span>
                  <span className="muted">{e.email}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </>
  );
}
