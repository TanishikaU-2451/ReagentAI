'use client';

import {
    ArrowDown,
    ChevronDown,
    Filter,
    Terminal,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

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
  DEBUG: 'text-sand-400',
  INFO: 'text-blue-700',
  WARNING: 'text-amber-700',
  ERROR: 'text-red-600',
};

const levelBgColors: Record<string, string> = {
  DEBUG: 'bg-sand-100',
  INFO: 'bg-blue-50',
  WARNING: 'bg-amber-50',
  ERROR: 'bg-red-50',
};

const levelBadgeColors: Record<string, string> = {
  DEBUG: 'bg-sand-200 text-sand-600',
  INFO: 'bg-blue-100 text-blue-700',
  WARNING: 'bg-amber-100 text-amber-700',
  ERROR: 'bg-red-100 text-red-700',
};

export default function LogsPanel({ logs }: LogsPanelProps) {
  const [filterLevel, setFilterLevel] = useState<LogLevel>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showFilter, setShowFilter] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Ensure we're on client side to prevent hydration issues
  useEffect(() => {
    setIsClient(true);
  }, []);

  const filteredLogs =
    filterLevel === 'ALL'
      ? logs
      : logs.filter((log) => log.level === filterLevel);

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
    if (!isClient) {
      // During SSR, return a placeholder to prevent hydration mismatch
      return '00:00:00.000';
    }
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
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-sand-200 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-sand-400" />
          <h3 className="text-xs font-semibold text-sand-600">
            Pipeline Logs
          </h3>
          <span className="text-[10px] text-sand-400 tabular-nums">
            ({filteredLogs.length})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Level counts */}
          <div className="flex items-center gap-1 mr-2">
            {levelCounts.ERROR > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-100 text-red-700 tabular-nums">
                {levelCounts.ERROR} errors
              </span>
            )}
            {levelCounts.WARNING > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 tabular-nums">
                {levelCounts.WARNING} warnings
              </span>
            )}
          </div>

          {/* Filter dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowFilter(!showFilter)}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-sand-500 hover:text-sand-800 hover:bg-sand-100 transition-colors"
            >
              <Filter className="w-3 h-3" />
              <span>{filterLevel}</span>
              <ChevronDown className="w-3 h-3" />
            </button>

            {showFilter && (
              <div className="absolute right-0 top-full mt-1 w-32 bg-white border border-sand-200 rounded-lg shadow-xl z-20 py-1">
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
                          ? 'bg-sand-100 text-sand-900'
                          : 'text-sand-600 hover:text-sand-900 hover:bg-sand-50'
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
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-sand-500 hover:text-sand-800 hover:bg-sand-100 transition-colors"
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
          <div className="flex items-center justify-center h-full text-sand-400">
            <p>No logs yet. Waiting for pipeline activity...</p>
          </div>
        ) : (
          <div className="p-2 space-y-px">
            {filteredLogs.map((log, idx) => (
              <div
                key={idx}
                className={`flex items-start gap-2 px-2 py-1 rounded ${levelBgColors[log.level] || ''}`}
              >
                <span className="text-sand-400 flex-shrink-0 tabular-nums w-20">
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
                  <span className="flex-shrink-0 text-sand-500 w-24 truncate">
                    [{log.agent}]
                  </span>
                )}
                <span
                  className={`flex-1 break-all ${
                    levelColors[log.level] || 'text-sand-600'
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
