'use client';

import { api } from '@/lib/api';
import {
    BookOpen,
    Bot,
    Loader2,
    MessageSquare,
    Send,
    Sparkles,
    User,
    X,
} from 'lucide-react';
import { FormEvent, useCallback, useEffect, useRef, useState } from 'react';

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
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-sand-200 flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-sand-100 border border-sand-300 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-sand-700" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-sand-900">Paper Q&A</h3>
            <p className="text-[10px] text-sand-400">
              Ask questions about the paper
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-sand-400 hover:text-sand-800 hover:bg-sand-100 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-8">
            <MessageSquare className="w-10 h-10 text-sand-300 mb-3" />
            <p className="text-sm text-sand-600 mb-1">
              Ask anything about the paper
            </p>
            <p className="text-xs text-sand-400 mb-6 max-w-[250px]">
              I can answer questions about the model architecture, training
              process, datasets, and more.
            </p>

            {/* Suggested questions */}
            <div className="space-y-2 w-full">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => sendMessage(q)}
                  className="w-full text-left px-3 py-2 rounded-lg border border-sand-200 text-xs text-sand-600 hover:text-sand-900 hover:border-sand-400 hover:bg-sand-50 transition-all"
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
            className={`flex gap-3 ${
              msg.role === 'user' ? 'flex-row-reverse' : ''
            }`}
          >
            {/* Avatar */}
            <div
              className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${
                msg.role === 'user'
                  ? 'bg-sand-800'
                  : 'bg-sand-100 border border-sand-300'
              }`}
            >
              {msg.role === 'user' ? (
                <User className="w-3.5 h-3.5 text-sand-100" />
              ) : (
                <Bot className="w-3.5 h-3.5 text-sand-700" />
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
                    ? 'bg-sand-800 text-sand-100'
                    : 'bg-sand-50 border border-sand-200 text-sand-800'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>

              {/* Citations */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 space-y-1.5">
                  <p className="text-[10px] text-sand-400 flex items-center gap-1">
                    <BookOpen className="w-3 h-3" />
                    Sources
                  </p>
                  {msg.citations.map((citation, idx) => (
                    <div
                      key={idx}
                      className="px-2.5 py-1.5 rounded-lg bg-sand-50 border border-sand-200 text-[11px]"
                    >
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="text-sand-700 font-medium">
                          {citation.section}
                        </span>
                        {citation.page && (
                          <span className="text-sand-400">
                            p.{citation.page}
                          </span>
                        )}
                      </div>
                      <p className="text-sand-500 line-clamp-2">
                        {citation.text}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              <p className="text-[10px] text-sand-400 mt-1">
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
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-lg bg-sand-100 border border-sand-300 flex items-center justify-center flex-shrink-0">
              <Bot className="w-3.5 h-3.5 text-sand-700" />
            </div>
            <div className="px-3 py-2 rounded-xl bg-sand-50 border border-sand-200">
              <div className="flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 text-sand-500 animate-spin" />
                <span className="text-xs text-sand-400">Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-sand-200 p-3 flex-shrink-0">
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
              className="w-full px-3 py-2.5 rounded-xl bg-sand-50 border border-sand-200 text-sm text-sand-900 placeholder-sand-400 focus:outline-none focus:border-sand-500 focus:ring-1 focus:ring-sand-300 resize-none disabled:opacity-50 transition-all"
              style={{ minHeight: '42px', maxHeight: '120px' }}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-2.5 rounded-xl bg-sand-800 text-sand-100 hover:bg-sand-700 disabled:opacity-30 disabled:hover:bg-sand-800 transition-all flex-shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
