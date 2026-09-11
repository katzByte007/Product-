import { useState } from 'react';
import AiTypeCards from '../components/ai-config/AiTypeCards';
import BetaConfigPanel from '../components/ai-config/BetaConfigPanel';
import DetectionConfigPanel from '../components/ai-config/DetectionConfigPanel';
import FrConfigPanel from '../components/ai-config/FrConfigPanel';
import { useAiConfigData } from '../hooks/useAiConfigData';

const SIMPLE_TYPES = ['headcount', 'entryexit', 'flapgate', 'workforce', 'intrusion', 'ppe', 'firesmoke', 'anpr'];

export default function AiConfigPage() {
  const [view, setView] = useState('types');
  const {
    cameras,
    configs,
    betaMeta,
    setBetaMeta,
    frActiveCameras,
    setFrActiveCameras,
    loading,
    reload,
    stats,
  } = useAiConfigData(view);

  const goBack = () => {
    setView('types');
    reload();
  };

  if (loading) {
    return <div className="page-loading">Loading AI configuration…</div>;
  }

  return (
    <div className="ai-config-page">
      {view === 'types' && <AiTypeCards stats={stats} onSelect={setView} />}

      {SIMPLE_TYPES.includes(view) && (
        <DetectionConfigPanel
          typeId={view}
          cameras={cameras}
          configs={configs}
          onBack={goBack}
          onReload={reload}
        />
      )}

      {view === 'beta' && (
        <BetaConfigPanel
          cameras={cameras}
          betaMeta={betaMeta}
          setBetaMeta={setBetaMeta}
          onBack={goBack}
          onReload={reload}
        />
      )}

      {view === 'fr' && (
        <FrConfigPanel
          cameras={cameras}
          frActiveCameras={frActiveCameras}
          setFrActiveCameras={setFrActiveCameras}
          onBack={goBack}
          onReload={reload}
        />
      )}
    </div>
  );
}
