'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  File,
  Folder,
  FolderOpen,
  Copy,
  Check,
  ChevronRight,
  ChevronDown,
  FileCode2,
  FileJson,
  FileText,
  Settings,
  Loader2,
} from 'lucide-react';
import { api } from '@/lib/api';

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
}

interface FileContent {
  path: string;
  content: string;
  language: string;
}

interface CodeViewerProps {
  projectId: string;
}

const fileIcons: Record<string, React.ElementType> = {
  py: FileCode2,
  ts: FileCode2,
  tsx: FileCode2,
  js: FileCode2,
  jsx: FileCode2,
  json: FileJson,
  yaml: Settings,
  yml: Settings,
  toml: Settings,
  cfg: Settings,
  ini: Settings,
  md: FileText,
  txt: FileText,
  rst: FileText,
};

function getFileIcon(filename: string): React.ElementType {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  return fileIcons[ext] || File;
}

function getLanguage(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const languageMap: Record<string, string> = {
    py: 'python',
    ts: 'typescript',
    tsx: 'typescript',
    js: 'javascript',
    jsx: 'javascript',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    toml: 'toml',
    md: 'markdown',
    sh: 'bash',
    bash: 'bash',
    dockerfile: 'dockerfile',
    cfg: 'ini',
    ini: 'ini',
    txt: 'text',
  };
  return languageMap[ext] || 'text';
}

function FileTreeNode({
  node,
  depth,
  selectedPath,
  onSelect,
}: {
  node: FileNode;
  depth: number;
  selectedPath: string;
  onSelect: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(depth < 2);
  const Icon = node.type === 'directory'
    ? expanded
      ? FolderOpen
      : Folder
    : getFileIcon(node.name);

  const isSelected = node.path === selectedPath;

  return (
    <div>
      <button
        onClick={() => {
          if (node.type === 'directory') {
            setExpanded(!expanded);
          } else {
            onSelect(node.path);
          }
        }}
        className={`w-full flex items-center gap-1.5 py-1 px-2 text-left text-xs rounded transition-colors ${
          isSelected
            ? 'bg-indigo-500/10 text-indigo-300'
            : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {node.type === 'directory' && (
          <span className="flex-shrink-0 w-3">
            {expanded ? (
              <ChevronDown className="w-3 h-3 text-gray-600" />
            ) : (
              <ChevronRight className="w-3 h-3 text-gray-600" />
            )}
          </span>
        )}
        {node.type !== 'directory' && <span className="w-3" />}
        <Icon
          className={`w-3.5 h-3.5 flex-shrink-0 ${
            node.type === 'directory'
              ? 'text-indigo-400/70'
              : isSelected
              ? 'text-indigo-400'
              : 'text-gray-500'
          }`}
        />
        <span className="truncate">{node.name}</span>
      </button>
      {node.type === 'directory' && expanded && node.children && (
        <div>
          {node.children.map((child, idx) => (
            <FileTreeNode
              key={idx}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function CodeViewer({ projectId }: CodeViewerProps) {
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [openFiles, setOpenFiles] = useState<FileContent[]>([]);
  const [activeFilePath, setActiveFilePath] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [treeLoading, setTreeLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  // Fetch file tree
  useEffect(() => {
    async function fetchTree() {
      setTreeLoading(true);
      try {
        const data = await api.getCodeFiles(projectId);
        setFileTree(data.files || []);
        // Auto-select first file
        if (data.files && data.files.length > 0) {
          const firstFile = findFirstFile(data.files);
          if (firstFile) {
            selectFile(firstFile);
          }
        }
      } catch (err) {
        console.error('Failed to fetch file tree:', err);
      } finally {
        setTreeLoading(false);
      }
    }
    if (projectId) {
      fetchTree();
    }
  }, [projectId]);

  function findFirstFile(nodes: FileNode[]): string | null {
    for (const node of nodes) {
      if (node.type === 'file') return node.path;
      if (node.children) {
        const found = findFirstFile(node.children);
        if (found) return found;
      }
    }
    return null;
  }

  const selectFile = useCallback(
    async (path: string) => {
      // Check if file is already open
      const existing = openFiles.find((f) => f.path === path);
      if (existing) {
        setActiveFilePath(path);
        return;
      }

      setLoading(true);
      try {
        const data = await api.getFileContent(projectId, path);
        const newFile: FileContent = {
          path,
          content: data.content || '',
          language: getLanguage(path),
        };
        setOpenFiles((prev) => [...prev, newFile]);
        setActiveFilePath(path);
      } catch (err) {
        console.error('Failed to fetch file:', err);
      } finally {
        setLoading(false);
      }
    },
    [projectId, openFiles]
  );

  const closeFile = useCallback(
    (path: string) => {
      setOpenFiles((prev) => prev.filter((f) => f.path !== path));
      if (activeFilePath === path) {
        const remaining = openFiles.filter((f) => f.path !== path);
        setActiveFilePath(remaining.length > 0 ? remaining[remaining.length - 1].path : '');
      }
    },
    [activeFilePath, openFiles]
  );

  const activeFile = openFiles.find((f) => f.path === activeFilePath);

  const handleCopy = useCallback(async () => {
    if (!activeFile) return;
    try {
      await navigator.clipboard.writeText(activeFile.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [activeFile]);

  const getFileName = (path: string) => {
    return path.split('/').pop() || path;
  };

  return (
    <div className="h-full flex overflow-hidden">
      {/* File Tree Sidebar */}
      <div className="w-56 bg-gray-900/30 border-r border-gray-800 overflow-y-auto scrollbar-thin flex-shrink-0">
        <div className="p-3 border-b border-gray-800">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Project Files
          </h3>
        </div>
        <div className="py-1">
          {treeLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-5 h-5 text-gray-600 animate-spin" />
            </div>
          ) : fileTree.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <File className="w-8 h-8 text-gray-700 mx-auto mb-2" />
              <p className="text-xs text-gray-600">
                No files generated yet
              </p>
            </div>
          ) : (
            fileTree.map((node, idx) => (
              <FileTreeNode
                key={idx}
                node={node}
                depth={0}
                selectedPath={activeFilePath}
                onSelect={selectFile}
              />
            ))
          )}
        </div>
      </div>

      {/* Code Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* File Tabs */}
        {openFiles.length > 0 && (
          <div className="flex items-center bg-gray-900/50 border-b border-gray-800 overflow-x-auto scrollbar-thin">
            {openFiles.map((file) => {
              const FileName = getFileIcon(file.path);
              return (
                <div
                  key={file.path}
                  className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-gray-800 cursor-pointer group min-w-fit ${
                    file.path === activeFilePath
                      ? 'bg-gray-800/50 text-white border-b-2 border-b-indigo-500'
                      : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30'
                  }`}
                  onClick={() => setActiveFilePath(file.path)}
                >
                  <FileName className="w-3 h-3 text-gray-500" />
                  <span>{getFileName(file.path)}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      closeFile(file.path);
                    }}
                    className="ml-1 p-0.5 rounded hover:bg-gray-700 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <span className="text-[10px] text-gray-500 hover:text-white">
                      x
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* Code Content */}
        <div className="flex-1 overflow-auto scrollbar-thin relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-950/50 z-10">
              <Loader2 className="w-6 h-6 text-indigo-400 animate-spin" />
            </div>
          )}
          {activeFile ? (
            <div className="relative">
              {/* Copy button */}
              <button
                onClick={handleCopy}
                className="absolute top-3 right-3 z-10 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-400 hover:text-white hover:border-gray-600 transition-all"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>

              {/* File path */}
              <div className="px-4 py-2 text-[11px] text-gray-600 bg-gray-900/30 border-b border-gray-800/50">
                {activeFile.path}
              </div>

              {/* Code display */}
              <pre className="p-4 text-sm font-mono leading-relaxed text-gray-300 overflow-auto">
                <code>
                  {activeFile.content.split('\n').map((line, i) => (
                    <div key={i} className="flex">
                      <span className="inline-block w-12 text-right pr-4 text-gray-700 select-none flex-shrink-0 text-xs leading-relaxed">
                        {i + 1}
                      </span>
                      <span className="flex-1 whitespace-pre-wrap break-all">
                        {line || ' '}
                      </span>
                    </div>
                  ))}
                </code>
              </pre>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-600">
              <div className="text-center">
                <FileCode2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm">Select a file to view its contents</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
