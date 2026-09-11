import { useEffect, useState } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { health, getAutoModeConfig } from '../api/endpoints';
import { useAuth } from '../context/AuthContext';
import AutoModePanel from './AutoModePanel';
import BrandLogo from './BrandLogo';

const links = [
  { to: '/', label: 'Cameras', end: true },
  { to: '/ai-config', label: 'AI Features' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/plant-map', label: 'Plant Map' },
  { to: '/alerts', label: 'Alerts' },
  { to: '/assigned-engineers', label: 'Assigned Engineers' },
  { to: '/system', label: 'System' },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [autoModeOpen, setAutoModeOpen] = useState(false);
  const [autoModeOn, setAutoModeOn] = useState(false);
  const [buildId, setBuildId] = useState('');

  useEffect(() => {
    health().then((h) => setBuildId(h.build || '')).catch(() => {});
  }, []);

  useEffect(() => {
    const sync = () => {
      getAutoModeConfig()
        .then((c) => setAutoModeOn(Boolean(c.enabled)))
        .catch(() => {});
    };
    sync();
    return () => {};
  }, [autoModeOpen]);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <div className="app-shell">
      <div className="ambient-bg" aria-hidden="true" />
      <header className="topbar glass-panel">
        <div className="brand">
          <BrandLogo variant="header" />
          {buildId ? <span className="build-stamp" title="Server build">{buildId}</span> : null}
        </div>
        <div className="topbar-right">
          <button
            type="button"
            className={`automode-top-btn neu-pill${autoModeOn ? ' on' : ''}`}
            onClick={() => setAutoModeOpen(true)}
            title="Autotrack — instruction watch across selected cameras"
          >
            Autotrack
          </button>
          <span className="user-pill neu-pill">
            {user?.username}
            <button type="button" className="link-btn" onClick={handleLogout}>Logout</button>
          </span>
          <nav className="main-nav">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => (isActive ? 'active' : undefined)}>
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="page-body"><Outlet /></main>
      <AutoModePanel open={autoModeOpen} onClose={() => setAutoModeOpen(false)} />
    </div>
  );
}
