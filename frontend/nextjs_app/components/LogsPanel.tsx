'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Terminal,
  ChevronDown,
  Filter,
  Trash2,
  ArrowDown,
} from 'lucide-react';

interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  agent?: string;
}

interface LogsPanelProps {
  logs: LogEntry[];
}

type LogLevel = 'ALL' | 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';

const levelColors: Record<string, string> = {
  DEBUG: 'text-gray-500',
  INFO: 'text-blue-400',
  WARNING: 'text-yellow-400',
  ERROR: 'text-red-400',
};

const levelBgColors: Record<string, string> = {
  DEBUG: 'bg-gray-500/10',
  INFO: 'bg-blue-500/10',
  WARNING: 'bg-yellow-500/10',
  ERROR: 'bg-red-500/10',
};

const levelBadgeColors: Record<string, string> = {
  DEBUG: 'bg-gray-700 text-gray-400',
  INFO: 'bg-blue-500/20 text-blue-400',
  WARNING: 'bg-yellow-500/20 text-yellow-400',
  ERROR: 'bg-red-500/20 text-red-400',
};

export default function LogsPanel({ logs }: LogsPanelProps) {
  const [filterLevel, setFilterLevel] = useState<LogLevel>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showFilter, setShowFilter] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const filteredLogs =
    filterLevel === 'ALL'
      ? logs
      : logs.filter((log) => log.level === filterLevel);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs.length, autoScroll]);

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(isAtBottom);
  }, []);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setAutoScroll(true);
  }, []);

  const formatTimestamp = (ts: string): string => {
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 3,
      });
    } catch {
      return ts;
    }
  };

  const levelCounts = {
    DEBUG: logs.filter((l) => l.level === 'DEBUG').length,
    INFO: logs.filter((l) => l.level === 'INFO').length,
    WARNING: logs.filter((l) => l.level === 'WARNING').length,
    ERROR: logs.filter((l) => l.level === 'ERROR').length,
  };

  return (
    <div className="h-full flex flex-col bg-gray-950">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-gray-500" />
          <h3 className="text-xs font-semibold text-gray-400">
            Pipeline Logs
          </h3>
          <span className="text-[10px] text-gray-600 tabular-nums">
            ({filteredLogs.length})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Level counts */}
          <div className="flex items-center gap-1 mr-2">
            {levelCounts.ERROR > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 tabular-nums">
                {levelCounts.ERROR} errors
              </span>
            )}
            {levelCounts.WARNING > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400 tabular-nums">
                {levelCounts.WARNING} warnings
              </span>
            )}
          </div>

          {/* Filter dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowFilter(!showFilter)}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-colors"
            >
              <Filter className="w-3 h-3" />
              <span>{filterLevel}</span>
              <ChevronDown className="w-3 h-3" />
            </button>

            {showFilter && (
              <div className="absolute right-0 top-full mt-1 w-32 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-20 py-1">
                {(['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR'] as LogLevel[]).map(
                  (level) => (
                    <button
                      key={level}
                      onClick={() => {
                        setFilterLevel(level);
                        setShowFilter(false);
                      }}
                      className={`w-full text-left px-3 py-1.5 text-xs transition-colors ${
                        filterLevel === level
                          ? 'bg-indigo-500/10 text-indigo-300'
                          : 'text-gray-400 hover:text-white hover:bg-gray-700'
                      }`}
                    >
                      {level}
                    </button>
                  )
                )}
              </div>
            )}
          </div>

          {/* Scroll to bottom */}
          {!autoScroll && (
            <button
              onClick={scrollToBottom}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-colors"
            >
              <ArrowDown className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto scrollbar-thin font-mono text-xs"
      >
        {filteredLogs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-700">
            <p>No logs yet. Waiting for pipeline activity...</p>
          </div>
        ) : (
          <div className="p-2 space-y-px">
            {filteredLogs.map((log, idx) => (
              <div
                key={idx}
                className={`flex items-start gap-2 px-2 py-1 rounded ${levelBgColors[log.level] || ''} animate-fade-in`}
              >
                <span className="text-gray-600 flex-shrink-0 tabular-nums w-20">
                  {formatTimestamp(log.timestamp)}
                </span>
                <span
                  className={`flex-shrink-0 w-16 text-center py-0 px-1 rounded text-[10px] font-semibold ${
                    levelBadgeColors[log.level] || ''
                  }`}
                >
                  {log.level}
                </span>
                {log.agent && (
                  <span className="flex-shrink-0 text-purple-400/70 w-24 truncate">
                    [{log.agent}]
                  </span>
                )}
                <span
                  className={`flex-1 break-all ${
                    levelColors[log.level] || 'text-gray-400'
                  }`}
                >
                  {log.message}
                </span>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>
    </div>
  );
}
