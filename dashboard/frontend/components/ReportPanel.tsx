"use client";

import { useEffect, useState, useRef } from "react";
import { FastForward } from "lucide-react";

interface ReportPanelProps {
  report: string;
  speed?: number; // ms per character
}

export default function ReportPanel({ report, speed = 8 }: ReportPanelProps) {
  const [displayedLength, setDisplayedLength] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!report) return;

    let idx = 0;
    const interval = setInterval(() => {
      idx += 1;
      const ch = report[idx - 1];
      if (ch === " " || ch === "\n") {
        idx += 1;
      }

      if (idx >= report.length) {
        setDisplayedLength(report.length);
        setIsComplete(true);
        clearInterval(interval);
      } else {
        setDisplayedLength(idx);
      }

      if (containerRef.current) {
        containerRef.current.scrollTop = containerRef.current.scrollHeight;
      }
    }, speed);

    return () => clearInterval(interval);
  }, [report, speed]);

  const handleSkip = () => {
    setDisplayedLength(report.length);
    setIsComplete(true);
  };

  return (
    <div className="relative h-full flex flex-col p-4">
      <div 
        className="font-mono text-sm leading-relaxed text-zinc-300 whitespace-pre-wrap overflow-y-auto pr-3 custom-scrollbar flex-1" 
        ref={containerRef}
      >
        {report.slice(0, displayedLength)}
        {!isComplete && (
          <span className="inline-block w-2 h-[1.1em] bg-medical-500 ml-1 align-text-bottom animate-pulse shadow-[0_0_8px_#14b8a6]" />
        )}
      </div>

      {!isComplete && (
        <button
          onClick={handleSkip}
          className="absolute bottom-4 right-6 bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-zinc-200 text-[10px] uppercase tracking-widest px-3 py-1.5 rounded-md transition-all font-semibold border border-white/10 flex items-center gap-1.5 backdrop-blur-md"
        >
          <FastForward className="w-3 h-3" />
          Skip
        </button>
      )}
    </div>
  );
}
