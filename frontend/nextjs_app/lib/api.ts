/**
 * ReagentAI API Client
 *
 * Typed functions for all backend API endpoints.
 * Base URL comes from NEXT_PUBLIC_API_URL env var, defaults to http://localhost:8000.
 */

const BASE_URL =
  typeof window !== 'undefined'
    ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
    : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

// ── Types ──────────────────────────────────────────────────────────────────────

export interface UploadResponse {
  project_id: string;
  message: string;
}

export interface ProjectInfo {
  id: string;
  title: string;
  status: string;
  paper_filename: string;
  stages: PipelineStage[];
  score?: number;
  logs?: LogEntry[];
}

export interface PipelineStage {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  agentName?: string;
  activityText?: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  agent?: string;
}

export interface FileTreeResponse {
  files: FileNode[];
}

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
}

export interface FileContentResponse {
  path: string;
  content: string;
  language?: string;
}

export interface DiagramsResponse {
  diagrams: DiagramData[];
}

export interface DiagramData {
  type: string;
  title: string;
  mermaidCode: string;
}

export interface ValidationResponse {
  summary: string;
  checks: ValidationCheck[];
}

export interface ValidationCheck {
  name: string;
  passed: boolean;
  message?: string;
}

export interface ScoreResponse {
  overall_score: number;
  categories: ScoreCategory[];
  feedback?: string;
}

export interface ScoreCategory {
  name: string;
  score: number;
  max: number;
}

export interface ChatResponse {
  message?: string;
  answer?: string;
  citations?: Citation[];
}

export interface Citation {
  section: string;
  page?: number;
  text: string;
}

// ── Error Handling ─────────────────────────────────────────────────────────────

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `API Error: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch {
      // Ignore JSON parse errors for error responses
    }
    throw new ApiError(errorMessage, response.status);
  }
  return response.json();
}

// ── API Functions ──────────────────────────────────────────────────────────────

export const api = {
  /**
   * Upload a PDF research paper for processing.
   */
  async uploadPaper(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse<UploadResponse>(response);
  },

  /**
   * Get project information and current status.
   */
  async getProject(projectId: string): Promise<ProjectInfo> {
    const response = await fetch(`${BASE_URL}/api/projects/${projectId}`);
    return handleResponse<ProjectInfo>(response);
  },

  /**
   * Get the file tree for a project's generated code.
   */
  async getCodeFiles(projectId: string): Promise<FileTreeResponse> {
    const response = await fetch(`${BASE_URL}/api/code/${projectId}`);
    return handleResponse<FileTreeResponse>(response);
  },

  /**
   * Get the content of a specific file.
   */
  async getFileContent(
    projectId: string,
    filePath: string
  ): Promise<FileContentResponse> {
    const encodedPath = encodeURIComponent(filePath);
    const response = await fetch(
      `${BASE_URL}/api/code/${projectId}/file?path=${encodedPath}`
    );
    return handleResponse<FileContentResponse>(response);
  },

  /**
   * Get architecture diagrams for a project.
   */
  async getDiagrams(projectId: string): Promise<DiagramsResponse> {
    const response = await fetch(`${BASE_URL}/api/diagrams/${projectId}`);
    return handleResponse<DiagramsResponse>(response);
  },

  /**
   * Get validation results for a project.
   */
  async getValidation(projectId: string): Promise<ValidationResponse> {
    const response = await fetch(`${BASE_URL}/api/validation/${projectId}`);
    return handleResponse<ValidationResponse>(response);
  },

  /**
   * Get quality score for a project.
   */
  async getScore(projectId: string): Promise<ScoreResponse> {
    const response = await fetch(`${BASE_URL}/api/score/${projectId}`);
    return handleResponse<ScoreResponse>(response);
  },

  /**
   * Send a chat message about the paper and receive an answer.
   */
  async sendChatMessage(
    projectId: string,
    message: string
  ): Promise<ChatResponse> {
    const response = await fetch(`${BASE_URL}/api/chat/${projectId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });
    return handleResponse<ChatResponse>(response);
  },

  /**
   * Download the generated project as a zip file.
   */
  async downloadProject(projectId: string): Promise<Blob> {
    const response = await fetch(`${BASE_URL}/api/download/${projectId}`);
    if (!response.ok) {
      throw new ApiError(
        `Download failed: ${response.status}`,
        response.status
      );
    }
    return response.blob();
  },
};

export default api;
