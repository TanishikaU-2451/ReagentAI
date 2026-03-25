'use client';

import { api } from '@/lib/api';
import {
    Check,
    ChevronDown,
    ChevronRight,
    Copy,
    File,
    FileCode2,
    FileJson,
    FileText,
    Folder,
    FolderOpen,
    Loader2,
    Settings,
} from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

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
            ? 'bg-sand-200 text-sand-900'
            : 'text-sand-600 hover:text-sand-900 hover:bg-sand-100'
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {node.type === 'directory' && (
          <span className="flex-shrink-0 w-3">
            {expanded ? (
              <ChevronDown className="w-3 h-3 text-sand-400" />
            ) : (
              <ChevronRight className="w-3 h-3 text-sand-400" />
            )}
          </span>
        )}
        {node.type !== 'directory' && <span className="w-3" />}
        <Icon
          className={`w-3.5 h-3.5 flex-shrink-0 ${
            node.type === 'directory'
              ? 'text-sand-500'
              : isSelected
              ? 'text-sand-700'
              : 'text-sand-400'
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

  useEffect(() => {
    async function fetchTree() {
      setTreeLoading(true);
      try {
        const data = await api.getCodeFiles(projectId);
        setFileTree(data.files || []);
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
      <div className="w-56 bg-sand-50 border-r border-sand-200 overflow-y-auto scrollbar-thin flex-shrink-0">
        <div className="p-3 border-b border-sand-200">
          <h3 className="text-xs font-semibold text-sand-500 uppercase tracking-wider">
            Project Files
          </h3>
        </div>
        <div className="py-1">
          {treeLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-5 h-5 text-sand-400 animate-spin" />
            </div>
          ) : fileTree.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <File className="w-8 h-8 text-sand-300 mx-auto mb-2" />
              <p className="text-xs text-sand-400">
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
          <div className="flex items-center bg-sand-50 border-b border-sand-200 overflow-x-auto scrollbar-thin">
            {openFiles.map((file) => {
              const FileName = getFileIcon(file.path);
              return (
                <div
                  key={file.path}
                  className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-sand-200 cursor-pointer group min-w-fit ${
                    file.path === activeFilePath
                      ? 'bg-white text-sand-900 border-b-2 border-b-sand-700'
                      : 'text-sand-500 hover:text-sand-800 hover:bg-sand-100'
                  }`}
                  onClick={() => setActiveFilePath(file.path)}
                >
                  <FileName className="w-3 h-3 text-sand-400" />
                  <span>{getFileName(file.path)}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      closeFile(file.path);
                    }}
                    className="ml-1 p-0.5 rounded hover:bg-sand-200 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <span className="text-[10px] text-sand-400 hover:text-sand-800">
                      x
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* Code Content */}
        <div className="flex-1 overflow-auto scrollbar-thin relative bg-white">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/70 z-10">
              <Loader2 className="w-6 h-6 text-sand-600 animate-spin" />
            </div>
          )}
          {activeFile ? (
            <div className="relative">
              {/* Copy button */}
              <button
                onClick={handleCopy}
                className="absolute top-3 right-3 z-10 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-sand-100 border border-sand-300 text-xs text-sand-600 hover:text-sand-900 hover:border-sand-400 transition-all"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-600" />
                    <span className="text-emerald-600">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>

              {/* File path */}
              <div className="px-4 py-2 text-[11px] text-sand-400 bg-sand-50 border-b border-sand-200">
                {activeFile.path}
              </div>

              {/* Code display */}
              <pre className="p-4 text-sm font-mono leading-relaxed text-sand-800 overflow-auto bg-sand-50/50">
                <code>
                  {activeFile.content.split('\n').map((line, i) => (
                    <div key={i} className="flex">
                      <span className="inline-block w-12 text-right pr-4 text-sand-400 select-none flex-shrink-0 text-xs leading-relaxed">
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
            <div className="flex items-center justify-center h-full text-sand-400">
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
