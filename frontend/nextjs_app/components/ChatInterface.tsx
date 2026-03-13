'use client';

import { useState, useRef, useEffect, useCallback, FormEvent } from 'react';
import {
  X,
  Send,
  Bot,
  User,
  Loader2,
  BookOpen,
  MessageSquare,
  Sparkles,
} from 'lucide-react';
import { api } from '@/lib/api';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  timestamp: Date;
}

interface Citation {
  section: string;
  page?: number;
  text: string;
}

interface ChatInterfaceProps {
  projectId: string;
  onClose: () => void;
}

const suggestedQuestions = [
  'What is the main model architecture?',
  'How is the training data preprocessed?',
  'What loss function is used?',
  'What are the key hyperparameters?',
];

export default function ChatInterface({
  projectId,
  onClose,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: text.trim(),
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setInput('');
      setIsLoading(true);

      try {
        const response = await api.sendChatMessage(projectId, text.trim());

        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.message || response.answer || '',
          citations: response.citations || [],
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err: any) {
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content:
            'Sorry, I encountered an error processing your question. Please try again.',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    [projectId, isLoading]
  );

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-950">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Paper Q&A</h3>
            <p className="text-[10px] text-gray-500">
              Ask questions about the paper
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-gray-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-8">
            <MessageSquare className="w-10 h-10 text-gray-800 mb-3" />
            <p className="text-sm text-gray-400 mb-1">
              Ask anything about the paper
            </p>
            <p className="text-xs text-gray-600 mb-6 max-w-[250px]">
              I can answer questions about the model architecture, training
              process, datasets, and more.
            </p>

            {/* Suggested questions */}
            <div className="space-y-2 w-full">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => sendMessage(q)}
                  className="w-full text-left px-3 py-2 rounded-lg border border-gray-800 text-xs text-gray-400 hover:text-white hover:border-gray-700 hover:bg-gray-900/50 transition-all"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 animate-fade-in ${
              msg.role === 'user' ? 'flex-row-reverse' : ''
            }`}
          >
            {/* Avatar */}
            <div
              className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${
                msg.role === 'user'
                  ? 'bg-indigo-600'
                  : 'bg-gray-800 border border-gray-700'
              }`}
            >
              {msg.role === 'user' ? (
                <User className="w-3.5 h-3.5 text-white" />
              ) : (
                <Bot className="w-3.5 h-3.5 text-indigo-400" />
              )}
            </div>

            {/* Message content */}
            <div
              className={`flex-1 max-w-[280px] ${
                msg.role === 'user' ? 'text-right' : ''
              }`}
            >
              <div
                className={`inline-block text-left px-3 py-2 rounded-xl text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-900 border border-gray-800 text-gray-300'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>

              {/* Citations */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 space-y-1.5">
                  <p className="text-[10px] text-gray-600 flex items-center gap-1">
                    <BookOpen className="w-3 h-3" />
                    Sources
                  </p>
                  {msg.citations.map((citation, idx) => (
                    <div
                      key={idx}
                      className="px-2.5 py-1.5 rounded-lg bg-gray-900/50 border border-gray-800/50 text-[11px]"
                    >
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="text-indigo-400 font-medium">
                          {citation.section}
                        </span>
                        {citation.page && (
                          <span className="text-gray-600">
                            p.{citation.page}
                          </span>
                        )}
                      </div>
                      <p className="text-gray-500 line-clamp-2">
                        {citation.text}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              <p className="text-[10px] text-gray-700 mt-1">
                {msg.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </p>
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-3 animate-fade-in">
            <div className="w-7 h-7 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center flex-shrink-0">
              <Bot className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <div className="px-3 py-2 rounded-xl bg-gray-900 border border-gray-800">
              <div className="flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
                <span className="text-xs text-gray-500">Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 p-3 flex-shrink-0">
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the paper..."
              rows={1}
              disabled={isLoading}
              className="w-full px-3 py-2.5 rounded-xl bg-gray-900 border border-gray-800 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 resize-none disabled:opacity-50 transition-all"
              style={{ minHeight: '42px', maxHeight: '120px' }}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-2.5 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-30 disabled:hover:bg-indigo-600 transition-all flex-shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
