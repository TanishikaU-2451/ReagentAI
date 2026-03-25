'use client';

import AgentProgressViewer from '@/components/AgentProgressViewer';
import ArchitectureDiagramViewer from '@/components/ArchitectureDiagramViewer';
import ChatInterface from '@/components/ChatInterface';
import CodeViewer from '@/components/CodeViewer';
import DownloadProject from '@/components/DownloadProject';
import LogsPanel from '@/components/LogsPanel';
import { api } from '@/lib/api';
import { useWebSocket } from '@/lib/websocket';
import {
    AlertCircle,
    BarChart3,
    CheckCircle2,
    Code2,
    FileText,
    GitFork,
    Loader2,
    MessageSquare,
    X,
} from 'lucide-react';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

interface PipelineStage {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  agentName?: string;
  activityText?: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

interface ProjectInfo {
  id: string;
  title: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  paperFilename: string;
  stages: PipelineStage[];
  score?: number;
}

interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  agent?: string;
}

type TabId = 'code' | 'diagrams' | 'validation' | 'score';

const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'code', label: 'Code', icon: Code2 },
  { id: 'diagrams', label: 'Diagrams', icon: GitFork },
  { id: 'validation', label: 'Validation', icon: CheckCircle2 },
  { id: 'score', label: 'Score', icon: BarChart3 },
];

export default function DashboardPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [activeTab, setActiveTab] = useState<TabId>('code');
  const [chatOpen, setChatOpen] = useState(false);
  const [project, setProject] = useState<ProjectInfo>({
    id: projectId,
    title: 'Initializing pipeline...',
    status: 'processing',
    paperFilename: '',
    stages: [],
  });
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [validationResults, setValidationResults] = useState<any>(null);
  const [scoreData, setScoreData] = useState<any>(null);

  const { messages, connectionStatus } = useWebSocket(projectId);

  // Handle 'latest' redirect to actual project ID
  useEffect(() => {
    if (projectId === 'latest') {
      fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/projects/latest`)
        .then(res => res.json())
        .then(data => {
          if (data.project_id) {
            router.replace(`/dashboard/${data.project_id}`);
          }
        })
        .catch(err => {
          console.error('Failed to resolve latest project ID:', err);
        });
    }
  }, [projectId, router]);

  // Initialize logs on client side only to prevent hydration mismatch
  useEffect(() => {
    setLogs([
      {
        timestamp: new Date().toISOString(),
        level: 'INFO',
        message: '✓ Paper uploaded successfully. Starting multi-agent pipeline...',
        agent: 'System',
      },
    ]);
  }, []); // Only run once on mount

  // Process WebSocket messages
  useEffect(() => {
    if (messages.length === 0) return;
    const latestMessage = messages[messages.length - 1];

    switch (latestMessage.type) {
      case 'project_info':
        setProject((prev) => ({
          ...prev,
          title: latestMessage.data.title || prev.title,
          status: latestMessage.data.status || prev.status,
          paperFilename:
            latestMessage.data.paperFilename || prev.paperFilename,
        }));
        break;

      case 'stage_update':
        setProject((prev) => {
          const stages = [...prev.stages];
          const stageIndex = stages.findIndex(
            (s) => s.name === latestMessage.data.stageName
          );
          const stageData: PipelineStage = {
            name: latestMessage.data.stageName,
            status: latestMessage.data.status,
            agentName: latestMessage.data.agentName,
            activityText: latestMessage.data.activityText,
            startedAt: latestMessage.data.startedAt,
            completedAt: latestMessage.data.completedAt,
            error: latestMessage.data.error,
          };
          if (stageIndex >= 0) {
            stages[stageIndex] = stageData;
          } else {
            stages.push(stageData);
          }
          return { ...prev, stages };
        });
        break;

      case 'log':
        setLogs((prev) => [
          ...prev,
          {
            timestamp: latestMessage.data.timestamp || new Date().toISOString(),
            level: latestMessage.data.level || 'INFO',
            message: latestMessage.data.message,
            agent: latestMessage.data.agent,
          },
        ]);
        break;

      case 'pipeline_complete':
        setProject((prev) => ({
          ...prev,
          status: 'completed',
          score: latestMessage.data.score,
        }));
        break;

      case 'pipeline_error':
        setProject((prev) => ({
          ...prev,
          status: 'error',
        }));
        break;

      case 'validation_result':
        setValidationResults(latestMessage.data);
        break;

      case 'score_result':
        setScoreData(latestMessage.data);
        break;
    }
  }, [messages]);

  // Fetch initial project info
  useEffect(() => {
    async function fetchProject() {
      try {
        const data = await api.getProject(projectId);
        if (data) {
          setProject({
            id: projectId,
            title: data.title || 'Untitled Paper',
            status: data.status || 'processing',
            paperFilename: data.paper_filename || '',
            stages: data.stages || [],
            score: data.score,
          });
          if (data.logs) {
            setLogs(data.logs);
          }
        }
      } catch (err) {
        console.error('Failed to fetch project:', err);
      }
    }
    if (projectId && projectId !== 'latest') {
      fetchProject();
    }
  }, [projectId]);

  // Fetch validation results when tab switches
  useEffect(() => {
    async function fetchValidation() {
      try {
        const data = await api.getValidation(projectId);
        setValidationResults(data);
      } catch (err) {
        console.error('Failed to fetch validation:', err);
      }
    }
    if (activeTab === 'validation' && !validationResults) {
      fetchValidation();
    }
  }, [activeTab, projectId, validationResults]);

  // Fetch score data when tab switches
  useEffect(() => {
    async function fetchScore() {
      try {
        const data = await api.getScore(projectId);
        setScoreData(data);
      } catch (err) {
        console.error('Failed to fetch score:', err);
      }
    }
    if (activeTab === 'score' && !scoreData) {
      fetchScore();
    }
  }, [activeTab, projectId, scoreData]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-emerald-600" />;
      case 'processing':
        return (
          <Loader2 className="w-4 h-4 text-sand-600 animate-spin" />
        );
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <FileText className="w-4 h-4 text-sand-400" />;
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-sand-200 flex-shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            {getStatusIcon(project.status)}
            <h1 className="text-lg font-semibold text-sand-900 truncate max-w-md">
              {project.title}
            </h1>
          </div>
          <span
            className={`text-xs px-2.5 py-1 rounded-full border ${
              project.status === 'completed'
                ? 'bg-emerald-50 border-emerald-300 text-emerald-700'
                : project.status === 'error'
                ? 'bg-red-50 border-red-300 text-red-600'
                : project.status === 'processing'
                ? 'bg-sand-100 border-sand-300 text-sand-700'
                : 'bg-sand-100 border-sand-200 text-sand-500'
            }`}
          >
            {project.status.charAt(0).toUpperCase() +
              project.status.slice(1)}
          </span>
          {project.score !== undefined && (
            <span className="text-xs text-sand-500">
              Score:{' '}
              <span className="text-sand-800 font-semibold">
                {project.score}/100
              </span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Connection indicator with helpful message */}
          <div className="flex items-center gap-1.5 text-xs">
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-emerald-500'
                  : connectionStatus === 'connecting'
                  ? 'bg-amber-500 animate-pulse'
                  : 'bg-sand-400'
              }`}
            />
            <span className={`${
              connectionStatus === 'connected'
                ? 'text-emerald-600'
                : connectionStatus === 'connecting'
                ? 'text-amber-600'
                : 'text-sand-500'
            }`}>
              {connectionStatus === 'connected'
                ? 'Live updates active'
                : connectionStatus === 'connecting'
                ? 'Connecting to pipeline...'
                : 'Checking for updates...'}
            </span>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center bg-sand-100 rounded-lg p-0.5 border border-sand-200">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-sand-800 text-sand-50 shadow-sm'
                      : 'text-sand-600 hover:text-sand-900 hover:bg-sand-200'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          <DownloadProject projectId={projectId} />

          <button
            onClick={() => setChatOpen(!chatOpen)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              chatOpen
                ? 'bg-sand-800 text-sand-50'
                : 'bg-sand-100 text-sand-700 hover:bg-sand-200 hover:text-sand-900 border border-sand-200'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            Chat
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Agent Progress Sidebar */}
        <div className="w-72 bg-white border-r border-sand-200 overflow-y-auto scrollbar-thin flex-shrink-0">
          <AgentProgressViewer stages={project.stages} />
        </div>

        {/* Center Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden bg-sand-50">
          {/* Main Tab Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'code' && (
              <CodeViewer projectId={projectId} />
            )}
            {activeTab === 'diagrams' && (
              <ArchitectureDiagramViewer projectId={projectId} />
            )}
            {activeTab === 'validation' && (
              <ValidationPanel results={validationResults} />
            )}
            {activeTab === 'score' && (
              <ScorePanel data={scoreData} />
            )}
          </div>

          {/* Logs Panel */}
          <div className="h-56 border-t border-sand-200 flex-shrink-0">
            <LogsPanel logs={logs} />
          </div>
        </div>

        {/* Chat Panel */}
        {chatOpen && (
          <div className="w-96 border-l border-sand-200 flex-shrink-0">
            <ChatInterface
              projectId={projectId}
              onClose={() => setChatOpen(false)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- Inline Sub-panels ---------- */

function ValidationPanel({ results }: { results: any }) {
  if (!results) {
    return (
      <div className="flex items-center justify-center h-full text-sand-400 bg-white">
        <div className="text-center">
          <CheckCircle2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Validation results will appear here</p>
          <p className="text-xs mt-1 text-sand-400">
            Results are generated after code generation completes
          </p>
        </div>
      </div>
    );
  }

  const checks = results.checks || [];

  return (
    <div className="h-full overflow-y-auto scrollbar-thin p-6 bg-white">
      <h2 className="text-lg font-semibold mb-4 text-sand-900">
        Validation Results
      </h2>
      {results.summary && (
        <div className="mb-6 p-4 rounded-xl bg-sand-50 border border-sand-200">
          <p className="text-sm text-sand-700">{results.summary}</p>
        </div>
      )}
      <div className="space-y-3">
        {checks.map((check: any, idx: number) => (
          <div
            key={idx}
            className={`p-4 rounded-xl border ${
              check.passed
                ? 'bg-emerald-50 border-emerald-200'
                : 'bg-red-50 border-red-200'
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              {check.passed ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              ) : (
                <X className="w-4 h-4 text-red-500" />
              )}
              <span
                className={`text-sm font-medium ${
                  check.passed ? 'text-emerald-700' : 'text-red-600'
                }`}
              >
                {check.name}
              </span>
            </div>
            {check.message && (
              <p className="text-xs text-sand-500 ml-6">{check.message}</p>
            )}
          </div>
        ))}
        {checks.length === 0 && (
          <p className="text-sm text-sand-400">No validation checks available.</p>
        )}
      </div>
    </div>
  );
}

function ScorePanel({ data }: { data: any }) {
  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-sand-400 bg-white">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm">Quality score will appear here</p>
          <p className="text-xs mt-1 text-sand-400">
            Scoring is performed after validation completes
          </p>
        </div>
      </div>
    );
  }

  const overall = data.overall_score ?? 0;
  const categories = data.categories || [];

  return (
    <div className="h-full overflow-y-auto scrollbar-thin p-6 bg-white">
      <h2 className="text-lg font-semibold mb-6 text-sand-900">Quality Score</h2>

      {/* Overall Score */}
      <div className="flex items-center justify-center mb-8">
        <div className="relative w-40 h-40">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
            <circle
              cx="60"
              cy="60"
              r="54"
              fill="none"
              stroke="#ebe3d7"
              strokeWidth="8"
            />
            <circle
              cx="60"
              cy="60"
              r="54"
              fill="none"
              stroke={
                overall >= 80
                  ? '#16a34a'
                  : overall >= 60
                  ? '#ca8a04'
                  : '#dc2626'
              }
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${(overall / 100) * 339.292} 339.292`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-bold text-sand-900">{overall}</span>
            <span className="text-xs text-sand-400">out of 100</span>
          </div>
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="space-y-4">
        {categories.map((cat: any, idx: number) => (
          <div key={idx}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-sm text-sand-700">{cat.name}</span>
              <span className="text-sm font-semibold text-sand-900">
                {cat.score}/{cat.max || 100}
              </span>
            </div>
            <div className="w-full bg-sand-200 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-500"
                style={{
                  width: `${(cat.score / (cat.max || 100)) * 100}%`,
                  backgroundColor:
                    cat.score / (cat.max || 100) >= 0.8
                      ? '#16a34a'
                      : cat.score / (cat.max || 100) >= 0.6
                      ? '#ca8a04'
                      : '#dc2626',
                }}
              />
            </div>
          </div>
        ))}
        {categories.length === 0 && (
          <p className="text-sm text-sand-400">No score breakdown available.</p>
        )}
      </div>

      {/* Feedback */}
      {data.feedback && (
        <div className="mt-6 p-4 rounded-xl bg-sand-50 border border-sand-200">
          <h3 className="text-sm font-medium text-sand-700 mb-2">Feedback</h3>
          <p className="text-xs text-sand-500 whitespace-pre-wrap">
            {data.feedback}
          </p>
        </div>
      )}
    </div>
  );
}
