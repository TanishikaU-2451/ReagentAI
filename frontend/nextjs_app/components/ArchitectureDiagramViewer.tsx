'use client';

import { api } from '@/lib/api';
import {
    Cpu,
    Database,
    GitFork,
    Loader2,
    Maximize2,
    RefreshCw,
    ZoomIn,
    ZoomOut,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

interface DiagramData {
  type: string;
  title: string;
  mermaidCode: string;
}

interface ArchitectureDiagramViewerProps {
  projectId: string;
}

const diagramTabs = [
  { id: 'model', label: 'Model Architecture', icon: Cpu },
  { id: 'training', label: 'Training Pipeline', icon: GitFork },
  { id: 'data_flow', label: 'Data Flow', icon: Database },
];

export default function ArchitectureDiagramViewer({
  projectId,
}: ArchitectureDiagramViewerProps) {
  const [activeTab, setActiveTab] = useState('model');
  const [diagrams, setDiagrams] = useState<Record<string, DiagramData>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [renderedSvg, setRenderedSvg] = useState<string>('');
  const [zoom, setZoom] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function fetchDiagrams() {
      setLoading(true);
      setError(null);
      try {
        const data = await api.getDiagrams(projectId);
        const diagramMap: Record<string, DiagramData> = {};
        if (data.diagrams) {
          for (const d of data.diagrams) {
            diagramMap[d.type] = d;
          }
        }
        setDiagrams(diagramMap);
      } catch (err: any) {
        setError(err.message || 'Failed to load diagrams');
      } finally {
        setLoading(false);
      }
    }
    if (projectId) {
      fetchDiagrams();
    }
  }, [projectId]);

  useEffect(() => {
    async function renderDiagram() {
      const diagram = diagrams[activeTab];
      if (!diagram || !diagram.mermaidCode) {
        setRenderedSvg('');
        return;
      }

      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: 'default',
          themeVariables: {
            darkMode: false,
            background: '#ffffff',
            primaryColor: '#f5f0ea',
            primaryTextColor: '#2a231c',
            primaryBorderColor: '#7d6a52',
            lineColor: '#b8a994',
            secondaryColor: '#faf8f5',
            tertiaryColor: '#f5f0ea',
            noteBkgColor: '#f5f0ea',
            noteTextColor: '#2a231c',
            noteBorderColor: '#ebe3d7',
            edgeLabelBackground: '#ffffff',
            clusterBkg: '#faf8f5',
            clusterBorder: '#ebe3d7',
            titleColor: '#2a231c',
          },
          flowchart: {
            curve: 'basis',
            padding: 20,
          },
          securityLevel: 'loose',
        });

        const uniqueId = `mermaid-${activeTab}-${Date.now()}`;
        const { svg } = await mermaid.render(uniqueId, diagram.mermaidCode);
        setRenderedSvg(svg);
      } catch (err: any) {
        console.error('Mermaid render error:', err);
        setRenderedSvg('');
        setError(`Failed to render ${activeTab} diagram: ${err.message}`);
      }
    }

    renderDiagram();
  }, [activeTab, diagrams]);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.2, 3));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.2, 0.3));
  const handleResetZoom = () => setZoom(1);

  const currentDiagram = diagrams[activeTab];

  return (
    <div className="h-full flex flex-col overflow-hidden bg-white">
      {/* Tab Bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-sand-200 flex-shrink-0">
        <div className="flex items-center gap-1">
          {diagramTabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-sand-200 text-sand-900 border border-sand-300'
                    : 'text-sand-500 hover:text-sand-800 hover:bg-sand-100'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded text-sand-400 hover:text-sand-800 hover:bg-sand-100 transition-colors"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-[10px] text-sand-400 w-10 text-center tabular-nums">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded text-sand-400 hover:text-sand-800 hover:bg-sand-100 transition-colors"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1.5 rounded text-sand-400 hover:text-sand-800 hover:bg-sand-100 transition-colors"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Diagram Content */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto scrollbar-thin flex items-center justify-center p-8 bg-sand-50/50"
      >
        {loading ? (
          <div className="text-center">
            <Loader2 className="w-8 h-8 text-sand-500 animate-spin mx-auto mb-3" />
            <p className="text-sm text-sand-400">Loading diagrams...</p>
          </div>
        ) : error && !renderedSvg ? (
          <div className="text-center max-w-md">
            <GitFork className="w-12 h-12 text-sand-300 mx-auto mb-3" />
            <p className="text-sm text-sand-500 mb-2">
              {error}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-1.5 text-xs text-sand-600 hover:text-sand-800 transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Retry
            </button>
          </div>
        ) : !currentDiagram ? (
          <div className="text-center">
            <GitFork className="w-12 h-12 text-sand-300 mx-auto mb-3" />
            <p className="text-sm text-sand-500">
              No {activeTab.replace('_', ' ')} diagram available
            </p>
            <p className="text-xs text-sand-400 mt-1">
              Diagrams will be generated during the pipeline execution
            </p>
          </div>
        ) : renderedSvg ? (
          <div
            className="transition-transform duration-200 mermaid"
            style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
            dangerouslySetInnerHTML={{ __html: renderedSvg }}
          />
        ) : (
          <div className="text-center">
            <Loader2 className="w-6 h-6 text-sand-400 animate-spin mx-auto mb-2" />
            <p className="text-xs text-sand-400">Rendering diagram...</p>
          </div>
        )}
      </div>

      {/* Mermaid source toggle */}
      {currentDiagram && currentDiagram.mermaidCode && (
        <MermaidSourceToggle code={currentDiagram.mermaidCode} />
      )}
    </div>
  );
}

function MermaidSourceToggle({ code }: { code: string }) {
  const [showSource, setShowSource] = useState(false);

  return (
    <div className="border-t border-sand-200 flex-shrink-0">
      <button
        onClick={() => setShowSource(!showSource)}
        className="w-full flex items-center justify-center gap-1.5 py-1.5 text-[10px] text-sand-400 hover:text-sand-700 transition-colors"
      >
        {showSource ? 'Hide' : 'Show'} Mermaid Source
      </button>
      {showSource && (
        <div className="max-h-40 overflow-auto scrollbar-thin bg-sand-50 p-3 border-t border-sand-200">
          <pre className="text-[11px] font-mono text-sand-600 whitespace-pre-wrap">
            {code}
          </pre>
        </div>
      )}
    </div>
  );
}
