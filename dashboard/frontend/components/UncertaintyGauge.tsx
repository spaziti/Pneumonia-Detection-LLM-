"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "../lib/utils";

interface UncertaintyGaugeProps {
  uncertainty: number;       // σ value, typically 0–0.3+
  meanProb: number;          // mean P(Pneumonia)
  mcPredictions?: number[];  // individual pass probabilities
}

export default function UncertaintyGauge({
  uncertainty,
  mcPredictions,
}: UncertaintyGaugeProps) {
  const [animatedAngle, setAnimatedAngle] = useState(0);

  const maxUnc = 0.35;
  const targetAngle = Math.min(uncertainty / maxUnc, 1) * 180;

  useEffect(() => {
    const timeout = setTimeout(() => setAnimatedAngle(targetAngle), 300);
    return () => clearTimeout(timeout);
  }, [targetAngle]);

  let verdict: string;
  let verdictClass: string;
  let verdictColor: string;
  
  if (uncertainty < 0.08) {
    verdict = "Auto-accept";
    verdictClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.15)]";
    verdictColor = "#34d399"; // emerald-400
  } else if (uncertainty < 0.2) {
    verdict = "Review Recommended";
    verdictClass = "bg-amber-500/10 text-amber-400 border border-amber-500/20 shadow-[0_0_15px_rgba(245,158,11,0.15)]";
    verdictColor = "#fbbf24"; // amber-400
  } else {
    verdict = "Flag for Human Review";
    verdictClass = "bg-rose-500/10 text-rose-400 border border-rose-500/20 shadow-[0_0_15px_rgba(244,63,94,0.15)]";
    verdictColor = "#fb7185"; // rose-400
  }

  const cx = 100, cy = 95;
  const r = 75;
  const startAngle = Math.PI;           
  const endAngle = startAngle - (animatedAngle / 180) * Math.PI;

  const startX = cx + r * Math.cos(startAngle);
  const startY = cy - r * Math.sin(startAngle);

  const needleAngle = startAngle - (animatedAngle / 180) * Math.PI;
  const needleX = cx + (r - 10) * Math.cos(needleAngle);
  const needleY = cy - (r - 10) * Math.sin(needleAngle);

  const endX = cx + r * Math.cos(endAngle);
  const endY = cy - r * Math.sin(endAngle);
  const largeArc = animatedAngle > 180 ? 1 : 0;
  const arcPath = `M ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY}`;

  const bgEndX = cx + r * Math.cos(0);
  const bgEndY = cy - r * Math.sin(0);
  const bgArcPath = `M ${startX} ${startY} A ${r} ${r} 0 0 1 ${bgEndX} ${bgEndY}`;

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-sm px-4">
      <div className="relative w-full max-w-[240px]">
        {/* Glow effect behind gauge */}
        <div 
          className="absolute inset-0 rounded-t-full opacity-20 blur-2xl transition-all duration-1000"
          style={{ backgroundColor: verdictColor }}
        />
        
        <svg className="w-full h-auto relative z-10" viewBox="0 0 200 120">
          <defs>
            <linearGradient id="gaugeGradDark" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#34d399" />
              <stop offset="35%" stopColor="#34d399" />
              <stop offset="55%" stopColor="#fbbf24" />
              <stop offset="80%" stopColor="#fb7185" />
              <stop offset="100%" stopColor="#fb7185" />
            </linearGradient>
            <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background Arc */}
          <path d={bgArcPath} fill="none" stroke="#27272a" strokeWidth="8" strokeLinecap="round" />
          
          {/* Animated Value Arc */}
          <path 
            d={arcPath} 
            fill="none" 
            stroke="url(#gaugeGradDark)" 
            strokeWidth="8" 
            strokeLinecap="round" 
            filter="url(#neonGlow)"
            className="transition-all duration-1000 ease-out" 
          />
          
          {/* Needle */}
          <line 
            x1={cx} y1={cy} 
            x2={needleX} y2={needleY} 
            stroke="#a1a1aa" 
            strokeWidth="3" 
            strokeLinecap="round" 
            className="transition-all duration-1000 ease-out drop-shadow-md" 
          />
          <circle cx={cx} cy={cy} r="6" fill="#a1a1aa" className="drop-shadow-md" />
          <circle cx={cx} cy={cy} r="2" fill="#27272a" />
        </svg>

        <div className="absolute bottom-0 left-0 right-0 flex justify-center translate-y-12">
          <div className="text-center flex flex-col items-center">
            <div className="font-mono text-3xl font-extrabold tracking-tight" style={{ color: verdictColor, textShadow: `0 0 15px ${verdictColor}40` }}>
              σ = {uncertainty.toFixed(4)}
            </div>
            <div className={cn("mt-3 px-4 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-widest", verdictClass)}>
              {verdict}
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between w-full mt-16 px-2 text-[10px] font-semibold text-zinc-500 uppercase tracking-widest">
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" /> Confident</div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-rose-400 shadow-[0_0_8px_#fb7185]" /> Unreliable</div>
      </div>

      {mcPredictions && mcPredictions.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="w-full mt-2 p-5 rounded-2xl bg-black/20 border border-white/5"
        >
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-3 flex items-center justify-between">
            <span>Distribution</span>
            <span className="font-mono text-zinc-600 bg-white/5 px-1.5 py-0.5 rounded">{mcPredictions.length} passes</span>
          </div>
          <div className="flex items-end gap-1 h-12">
            {(() => {
              const bins = new Array(20).fill(0);
              mcPredictions.forEach(p => {
                const idx = Math.min(Math.floor(p * 20), 19);
                bins[idx]++;
              });
              const maxBin = Math.max(...bins, 1);
              return bins.map((count, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t transition-all duration-700 ease-out hover:opacity-80"
                  style={{
                    height: `${(count / maxBin) * 100}%`,
                    minHeight: count > 0 ? 3 : 0,
                    background: i < 10 ? "#34d399" : "#fb7185", // emerald-400 : rose-400
                    boxShadow: count > 0 ? `0 0 10px ${i < 10 ? '#34d399' : '#fb7185'}30` : 'none'
                  }}
                />
              ));
            })()}
          </div>
          <div className="flex justify-between text-[9px] font-semibold text-zinc-600 uppercase tracking-wider mt-2">
            <span>Normal</span>
            <span>Pneumonia</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
