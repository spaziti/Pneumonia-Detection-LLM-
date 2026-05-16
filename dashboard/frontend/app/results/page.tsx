"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, Variants } from "framer-motion";
import { Activity, Target, AlertTriangle, FileText, Flame, History, ArrowLeft, Thermometer, HeartPulse, Wind } from "lucide-react";
import GradcamViewer from "../../components/GradcamViewer";
import UncertaintyGauge from "../../components/UncertaintyGauge";
import ReportPanel from "../../components/ReportPanel";
import { cn } from "../../lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface AnalysisResult {
  prediction: string;
  prediction_idx: number;
  confidence: number;
  uncertainty: number;
  mean_prob: number;
  mc_predictions: number[];
  spatial_info: {
    primary_region: string;
    primary_score: number;
    secondary_region: string | null;
    secondary_score: number | null;
    overall_spread: number;
    region_scores: Record<string, number>;
  };
  gradcam_base64: string;
  original_base64: string;
  report: string;
  ehr_data: Record<string, number> | null;
  image_name: string;
}

interface HistoryRecord {
  id: number;
  timestamp: string;
  image_name: string;
  prediction: string;
  confidence: number;
  uncertainty: number;
}

export default function ResultsPage() {
  const router = useRouter();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.records || []);
      }
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    const stored = sessionStorage.getItem("analysisResult");
    if (!stored) {
      router.push("/");
    } else {
      setResult(JSON.parse(stored));
      fetchHistory();
    }
  }, [router]);

  if (!result) return null;

  const isPneumonia = result.prediction === "PNEUMONIA";
  const confPct = (result.confidence * 100).toFixed(1);
  const circumference = 2 * Math.PI * 56;
  const offset = circumference * (1 - result.confidence);

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-[#09090b]">
      {/* Background gradients */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-medical-900/10 via-[#09090b] to-[#09090b] -z-10 fixed" />

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-5 border-b border-white/5 sticky top-0 z-40 bg-[#09090b]/80 backdrop-blur-xl">
        <div 
          className="flex items-center gap-3 font-bold text-xl text-zinc-100 tracking-tight cursor-pointer hover:text-medical-400 transition-colors" 
          onClick={() => router.push("/")}
        >
          <div className="w-10 h-10 rounded-xl bg-medical-500/10 text-medical-400 flex items-center justify-center border border-medical-500/20">
            <Activity className="w-5 h-5" />
          </div>
          <span>Pneumo<span className="text-medical-400">Scan</span></span>
        </div>
        <nav>
          <button className="btn-ghost text-sm" onClick={() => router.push("/")}>
            <ArrowLeft className="w-4 h-4" />
            New Analysis
          </button>
        </nav>
      </header>

      {/* Main Content - Bento Grid */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 sm:p-6 lg:p-8">
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-12 gap-6"
          variants={containerVariants}
          initial="hidden"
          animate="show"
        >
          
          {/* --- 1. Prediction Card (Bento span: XL-4, MD-1) --- */}
          <motion.div variants={itemVariants} className="glass-card xl:col-span-4 flex flex-col justify-center">
            <div className="card-header">
              <Target className="w-4 h-4 text-medical-400" />
              <span>Classification</span>
            </div>
            <div className="card-body flex items-center gap-6 xl:flex-col xl:items-start 2xl:flex-row 2xl:items-center py-8">
              <div className="relative w-32 h-32 shrink-0">
                <svg viewBox="0 0 120 120" className="-rotate-90 w-full h-full drop-shadow-[0_0_15px_rgba(255,255,255,0.1)]">
                  <circle className="fill-none stroke-white/5" strokeWidth="8" cx="60" cy="60" r="56" />
                  <motion.circle
                    className="fill-none drop-shadow-md"
                    strokeWidth="8"
                    strokeLinecap="round"
                    cx="60"
                    cy="60"
                    r="56"
                    stroke={isPneumonia ? "#f43f5e" : "#10b981"} // rose-500 or emerald-500
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <div className="text-2xl font-extrabold text-zinc-100" style={{ color: isPneumonia ? "#f43f5e" : "#10b981" }}>
                    {confPct}%
                  </div>
                  <div className="text-[9px] uppercase tracking-widest text-zinc-500 font-semibold mt-1">Confidence</div>
                </div>
              </div>

              <div className="flex-1">
                <div className="text-sm font-semibold text-zinc-500 uppercase tracking-widest mb-1">Result</div>
                <div className={cn(
                  "text-3xl sm:text-4xl font-extrabold tracking-tight mb-4", 
                  isPneumonia ? "text-rose-500 drop-shadow-[0_0_15px_rgba(244,63,94,0.3)]" : "text-emerald-500 drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                )}>
                  {result.prediction}
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center border-b border-white/5 pb-2">
                    <span className="text-zinc-500">Primary Region</span>
                    <span className="font-mono text-zinc-200">{result.spatial_info.primary_region}</span>
                  </div>
                  {result.spatial_info.secondary_region && (
                    <div className="flex justify-between items-center border-b border-white/5 pb-2">
                      <span className="text-zinc-500">Secondary</span>
                      <span className="font-mono text-zinc-200">{result.spatial_info.secondary_region}</span>
                    </div>
                  )}
                  <div className="flex justify-between items-center">
                    <span className="text-zinc-500">Spread</span>
                    <span className="font-mono text-zinc-200">{(result.spatial_info.overall_spread * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* --- 2. Grad-CAM Viewer (Bento span: XL-8, MD-1) --- */}
          <motion.div variants={itemVariants} className="glass-card xl:col-span-8 overflow-hidden flex flex-col">
            <div className="card-header">
              <Flame className="w-4 h-4 text-orange-500" />
              <span>Explainable AI Heatmap</span>
            </div>
            <div className="card-body p-0 flex-1">
              <GradcamViewer
                originalBase64={result.original_base64}
                gradcamBase64={result.gradcam_base64}
                spatialInfo={result.spatial_info}
              />
            </div>
          </motion.div>

          {/* --- 3. Uncertainty Gauge (Bento span: XL-4) --- */}
          <motion.div variants={itemVariants} className="glass-card xl:col-span-4 flex flex-col">
            <div className="card-header">
              <AlertTriangle className="w-4 h-4 text-amber-500" />
              <span>Diagnostic Uncertainty (MC Dropout)</span>
            </div>
            <div className="card-body flex justify-center items-center py-10 flex-1">
              <UncertaintyGauge
                uncertainty={result.uncertainty}
                meanProb={result.mean_prob}
                mcPredictions={result.mc_predictions}
              />
            </div>
          </motion.div>

          {/* --- 4. Report Panel (Bento span: XL-5) --- */}
          <motion.div variants={itemVariants} className="glass-card xl:col-span-5 flex flex-col">
            <div className="card-header">
              <FileText className="w-4 h-4 text-blue-400" />
              <span>Automated Radiology Report</span>
            </div>
            <div className="card-body flex-1 overflow-y-auto custom-scrollbar">
              <ReportPanel report={result.report} />
            </div>
          </motion.div>

          {/* --- 5. Sidebar Stack: EHR + History (Bento span: XL-3) --- */}
          <div className="xl:col-span-3 flex flex-col gap-6">
            
            {/* EHR Data */}
            {result.ehr_data && (
              <motion.div variants={itemVariants} className="glass-card flex-1">
                <div className="card-header">
                  <Activity className="w-4 h-4 text-purple-400" />
                  <span>Clinical EHR Data</span>
                </div>
                <div className="card-body">
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(result.ehr_data).map(([key, val]) => (
                      <div key={key} className="p-3 rounded-xl bg-black/20 border border-white/5 hover:border-white/10 transition-colors">
                        <div className="text-[9px] text-zinc-500 uppercase tracking-widest font-semibold mb-1 truncate flex items-center gap-1.5">
                          {key.includes('temp') && <Thermometer className="w-3 h-3" />}
                          {key.includes('heart') && <HeartPulse className="w-3 h-3" />}
                          {key.includes('resp') && <Wind className="w-3 h-3" />}
                          {key.replace(/_/g, " ")}
                        </div>
                        <div className="text-lg font-bold font-mono text-zinc-200">
                          {typeof val === "number" ? val.toFixed(1) : val}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* History */}
            <motion.div variants={itemVariants} className="glass-card flex-1 flex flex-col max-h-[400px]">
              <div className="card-header shrink-0">
                <History className="w-4 h-4 text-zinc-400" />
                <span>Recent Analyses</span>
              </div>
              <div className="card-body overflow-y-auto p-4 flex-1 custom-scrollbar">
                {history.length === 0 ? (
                  <div className="text-center py-8 text-zinc-600 text-sm font-mono">No previous records.</div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {history.slice(0, 5).map((item) => (
                      <div className="flex items-center gap-3 p-3 rounded-xl bg-black/20 border border-white/5 hover:bg-white/5 hover:border-white/10 transition-colors" key={item.id}>
                        <div className={cn(
                          "w-2 h-2 rounded-full shrink-0 shadow-[0_0_8px_currentColor]", 
                          item.prediction === "PNEUMONIA" ? "bg-rose-500 text-rose-500" : "bg-emerald-500 text-emerald-500"
                        )} />
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-semibold text-zinc-300 truncate">{item.image_name}</div>
                          <div className="text-[10px] text-zinc-500 mt-0.5">{new Date(item.timestamp).toLocaleString()}</div>
                        </div>
                        <div className="text-[10px] font-mono font-bold text-zinc-400 bg-white/5 px-2 py-1 rounded-md border border-white/10">
                          {(item.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </div>

        </motion.div>
      </main>
    </div>
  );
}
