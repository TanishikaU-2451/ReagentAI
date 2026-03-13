'use client';

import {
  FileSearch,
  Brain,
  Code2,
  TestTube2,
  BarChart3,
  GitFork,
  CheckCircle2,
  XCircle,
  Loader2,
  Circle,
  Cpu,
  Database,
  BookOpen,
  Wrench,
} from 'lucide-react';

interface PipelineStage {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  agentName?: string;
  activityText?: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

interface AgentProgressViewerProps {
  stages: PipelineStage[];
}

const defaultStages: PipelineStage[] = [
  { name: 'Paper Parsing', status: 'pending' },
  { name: 'Paper Analysis', status: 'pending' },
  { name: 'Architecture Design', status: 'pending' },
  { name: 'Code Generation', status: 'pending' },
  { name: 'Data Pipeline', status: 'pending' },
  { name: 'Training Setup', status: 'pending' },
  { name: 'Diagram Generation', status: 'pending' },
  { name: 'Validation', status: 'pending' },
  { name: 'Scoring', status: 'pending' },
];

const stageIcons: Record<string, React.ElementType> = {
  'Paper Parsing': FileSearch,
  'Paper Analysis': BookOpen,
  'Architecture Design': Brain,
  'Code Generation': Code2,
  'Data Pipeline': Database,
  'Training Setup': Cpu,
  'Diagram Generation': GitFork,
  'Validation': TestTube2,
  'Scoring': BarChart3,
};

export default function AgentProgressViewer({
  stages,
}: AgentProgressViewerProps) {
  const displayStages =
    stages.length > 0 ? stages : defaultStages;

  const completedCount = displayStages.filter(
    (s) => s.status === 'completed'
  ).length;
  const totalCount = displayStages.length;
  const progressPercent =
    totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

  const runningStage = displayStages.find((s) => s.status === 'running');

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-white mb-3">
          Pipeline Progress
        </h2>
        <div className="flex items-center gap-3 mb-2">
          <div className="flex-1 bg-gray-800 rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full bg-indigo-500 transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <span className="text-xs text-gray-400 tabular-nums flex-shrink-0">
            {completedCount}/{totalCount}
          </span>
        </div>
        {runningStage && (
          <div className="flex items-center gap-2 mt-2 px-2 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
            <Loader2 className="w-3 h-3 text-indigo-400 animate-spin flex-shrink-0" />
            <span className="text-xs text-indigo-300 truncate">
              {runningStage.activityText ||
                `Running ${runningStage.name}...`}
            </span>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-[15px] top-2 bottom-2 w-px bg-gray-800" />

          <div className="space-y-1">
            {displayStages.map((stage, index) => {
              const Icon = stageIcons[stage.name] || Wrench;
              return (
                <div key={index} className="relative flex items-start gap-3">
                  {/* Status indicator */}
                  <div className="relative z-10 flex-shrink-0 mt-0.5">
                    {stage.status === 'completed' && (
                      <div className="w-[30px] h-[30px] rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      </div>
                    )}
                    {stage.status === 'running' && (
                      <div className="w-[30px] h-[30px] rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center">
                        <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                      </div>
                    )}
                    {stage.status === 'error' && (
                      <div className="w-[30px] h-[30px] rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center">
                        <XCircle className="w-4 h-4 text-red-400" />
                      </div>
                    )}
                    {stage.status === 'pending' && (
                      <div className="w-[30px] h-[30px] rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center">
                        <Circle className="w-3 h-3 text-gray-600" />
                      </div>
                    )}
                  </div>

                  {/* Stage info */}
                  <div
                    className={`flex-1 py-2 px-3 rounded-lg transition-all ${
                      stage.status === 'running'
                        ? 'bg-indigo-500/5 border border-indigo-500/10'
                        : stage.status === 'error'
                        ? 'bg-red-500/5 border border-red-500/10'
                        : ''
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Icon
                        className={`w-3.5 h-3.5 ${
                          stage.status === 'completed'
                            ? 'text-emerald-400'
                            : stage.status === 'running'
                            ? 'text-indigo-400'
                            : stage.status === 'error'
                            ? 'text-red-400'
                            : 'text-gray-600'
                        }`}
                      />
                      <span
                        className={`text-sm font-medium ${
                          stage.status === 'completed'
                            ? 'text-gray-300'
                            : stage.status === 'running'
                            ? 'text-white'
                            : stage.status === 'error'
                            ? 'text-red-300'
                            : 'text-gray-500'
                        }`}
                      >
                        {stage.name}
                      </span>
                    </div>

                    {stage.agentName && (
                      <p className="text-[11px] text-gray-500 mt-0.5 ml-5.5">
                        Agent: {stage.agentName}
                      </p>
                    )}

                    {stage.status === 'running' && stage.activityText && (
                      <p className="text-[11px] text-indigo-400/70 mt-1 ml-5.5 truncate">
                        {stage.activityText}
                      </p>
                    )}

                    {stage.status === 'error' && stage.error && (
                      <p className="text-[11px] text-red-400/70 mt-1 ml-5.5 truncate">
                        {stage.error}
                      </p>
                    )}

                    {stage.completedAt && (
                      <p className="text-[10px] text-gray-600 mt-0.5 ml-5.5">
                        {new Date(stage.completedAt).toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
