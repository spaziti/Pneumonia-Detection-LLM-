"use client";

import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";

interface GradcamViewerProps {
  originalBase64: string;
  gradcamBase64: string;
  spatialInfo: {
    primary_region: string;
    primary_score: number;
    secondary_region: string | null;
    secondary_score: number | null;
    overall_spread: number;
    region_scores: Record<string, number>;
  };
}

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function GradcamViewer({
  originalBase64,
  gradcamBase64: initialGradcam,
  spatialInfo,
}: GradcamViewerProps) {
  const [alpha, setAlpha] = useState(0.4);
  const [gradcamSrc, setGradcamSrc] = useState(initialGradcam);
  const [isUpdating, setIsUpdating] = useState(false);

  useEffect(() => {
    setGradcamSrc(initialGradcam);
    setAlpha(0.4);
  }, [initialGradcam]);

  const updateThreshold = useCallback(
    async (newAlpha: number) => {
      setIsUpdating(true);
      try {
        const form = new FormData();
        form.append("alpha", String(newAlpha));
        const res = await fetch(`${API}/api/gradcam-threshold`, {
          method: "POST",
          body: form,
        });
        if (res.ok) {
          const data = await res.json();
          setGradcamSrc(data.gradcam_base64);
        }
      } catch {
        /* keep current image */
      } finally {
        setIsUpdating(false);
      }
    },
    []
  );

  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setAlpha(val);
  };

  const handleSliderEnd = () => {
    updateThreshold(alpha);
  };

  return (
    <div className="flex flex-col h-full gap-4 p-4">
      <div className="grid grid-cols-2 gap-4 flex-1">
        <div className="flex flex-col items-center justify-center">
          <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/40 p-1 w-full flex-1 flex items-center justify-center min-h-[200px]">
            <img
              src={`data:image/png;base64,${originalBase64}`}
              alt="Original X-ray"
              className="max-h-full max-w-full rounded-lg object-contain opacity-80"
            />
          </div>
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mt-3">Original</div>
        </div>
        <div className="flex flex-col items-center justify-center">
          <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/40 p-1 w-full flex-1 flex items-center justify-center min-h-[200px]">
            <img
              src={`data:image/png;base64,${gradcamSrc}`}
              alt="Grad-CAM heatmap overlay"
              className={`max-h-full max-w-full rounded-lg object-contain transition-opacity duration-300 ${isUpdating ? "opacity-40 filter blur-sm" : "opacity-100"}`}
            />
            {isUpdating && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-6 h-6 border-2 border-medical-500/30 border-t-medical-500 rounded-full animate-spin" />
              </div>
            )}
          </div>
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mt-3">
            Grad-CAM <span className="font-mono bg-white/5 px-1.5 py-0.5 rounded text-zinc-400 border border-white/5 ml-1">α={alpha.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 bg-black/20 p-3 rounded-xl border border-white/5">
        <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider whitespace-nowrap">Heatmap α</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={alpha}
          onChange={handleSlider}
          onMouseUp={handleSliderEnd}
          onTouchEnd={handleSliderEnd}
          className="flex-1 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-medical-500 hover:accent-medical-400 transition-all"
        />
        <span className="font-mono text-sm font-bold text-medical-400 min-w-[40px] text-right">{alpha.toFixed(2)}</span>
      </div>

      {/* Region breakdown */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {Object.entries(spatialInfo.region_scores).map(([region, score]) => {
          const isPrimary = region === spatialInfo.primary_region;
          return (
            <motion.div
              key={region}
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className={`flex flex-col justify-center items-center p-2 rounded-xl text-xs font-medium border transition-colors ${
                isPrimary 
                ? "bg-medical-500/10 border-medical-500/30 text-medical-300 shadow-[0_0_10px_rgba(20,184,166,0.1)]" 
                : "bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10"
              }`}
            >
              <span className="text-[10px] uppercase tracking-wider mb-1">{region}</span>
              <span className="font-mono font-bold text-sm">
                {(score * 100).toFixed(0)}%
              </span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
