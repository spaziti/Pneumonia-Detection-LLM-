"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { ScanFace, UploadCloud, ChevronDown, Activity, Sparkles, FileImage } from "lucide-react";
import { cn } from "../lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EHR_FIELDS = [
  { key: "age", label: "Age", placeholder: "45", type: "number" },
  { key: "temperature", label: "Temp (°C)", placeholder: "37.0", type: "number" },
  { key: "heart_rate", label: "Heart Rate", placeholder: "80", type: "number" },
  { key: "wbc_count", label: "WBC (×10³/μL)", placeholder: "8.0", type: "number" },
  { key: "respiratory_rate", label: "Resp. Rate", placeholder: "18", type: "number" },
  { key: "cough_duration_days", label: "Cough (days)", placeholder: "3", type: "number" },
  { key: "oxygen_saturation", label: "SpO₂ (%)", placeholder: "97", type: "number" },
];

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showEHR, setShowEHR] = useState(false);
  const [ehr, setEhr] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");

  const handleFile = (f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  }, []);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);

    const steps = [
      "Initializing AI models...",
      "Preprocessing X-ray image...",
      "Running HATR classification...",
      "Generating Grad-CAM heatmap...",
      "Computing MC Dropout uncertainty...",
      "Synthesizing radiology report...",
    ];
    let stepIdx = 0;
    setLoadingStep(steps[0]);
    const stepInterval = setInterval(() => {
      stepIdx = Math.min(stepIdx + 1, steps.length - 1);
      setLoadingStep(steps[stepIdx]);
    }, 1500);

    try {
      const form = new FormData();
      form.append("file", file);

      const ehrData: Record<string, number> = {};
      let hasEhr = false;
      for (const field of EHR_FIELDS) {
        if (ehr[field.key]) {
          ehrData[field.key] = parseFloat(ehr[field.key]);
          hasEhr = true;
        }
      }
      if (hasEhr) {
        form.append("ehr_json", JSON.stringify(ehrData));
      }

      const res = await fetch(`${API}/api/predict`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();

      sessionStorage.setItem("analysisResult", JSON.stringify(data));
      router.push("/results");
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Is the backend running on localhost:8000?");
      setLoading(false);
    } finally {
      clearInterval(stepInterval);
    }
  };

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background gradients */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-medical-900/20 via-[#09090b] to-[#09090b] -z-10" />

      {/* Loading Overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-[#09090b]/80 backdrop-blur-xl z-50 flex flex-col items-center justify-center gap-8"
          >
            <div className="relative w-32 h-32 flex items-center justify-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 rounded-full border-t-2 border-medical-500 border-r-2 border-r-transparent border-l-2 border-l-transparent"
              />
              <motion.div
                animate={{ rotate: -360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="absolute inset-4 rounded-full border-b-2 border-medical-300 border-r-2 border-r-transparent border-l-2 border-l-transparent opacity-70"
              />
              <ScanFace className="w-10 h-10 text-medical-400 animate-pulse" />
            </div>
            
            <div className="text-center space-y-3">
              <h3 className="text-2xl font-semibold text-zinc-100 tracking-tight">Analyzing Imaging</h3>
              <motion.p
                key={loadingStep}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-medical-400/80 font-mono text-sm max-w-sm"
              >
                {loadingStep}
              </motion.p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-6 sm:px-10 z-40">
        <div className="flex items-center gap-3 font-bold text-xl text-zinc-100 tracking-tight">
          <div className="w-10 h-10 rounded-xl bg-medical-500/10 text-medical-400 flex items-center justify-center border border-medical-500/20 shadow-[0_0_15px_rgba(20,184,166,0.15)]">
            <ScanFace className="w-5 h-5" />
          </div>
          <span>Pneumo<span className="text-medical-400">Scan</span></span>
        </div>
        <nav>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/5 text-zinc-300 text-xs font-semibold tracking-wide uppercase border border-white/10 backdrop-blur-md">
            <Sparkles className="w-3 h-3 text-medical-400" />
            HATR Hybrid Engine
          </span>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center w-full max-w-4xl mx-auto px-6 py-12 gap-10 z-10">
        
        {/* Hero Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center space-y-4"
        >
          <h1 className="text-5xl sm:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-br from-zinc-100 via-zinc-300 to-zinc-600 tracking-tight leading-tight">
            Intelligent <br className="hidden sm:block" /> 
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-medical-300 to-medical-500">
              Pneumonia Detection
            </span>
          </h1>
          <p className="text-lg text-zinc-400 max-w-2xl mx-auto leading-relaxed">
            Upload a chest X-ray to leverage our clinical-grade HATR model. 
            Experience real-time explainability, uncertainty quantification, and automated radiology reports.
          </p>
        </motion.div>

        {/* Upload Zone */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full max-w-2xl"
        >
          <motion.div
            className={cn(
              "relative group w-full border-2 border-dashed rounded-3xl p-10 text-center cursor-pointer transition-all duration-300 overflow-hidden",
              isDragging 
                ? "border-medical-400 bg-medical-500/10 shadow-[0_0_30px_rgba(20,184,166,0.15)]" 
                : "border-white/10 bg-white/[0.02] hover:border-medical-500/50 hover:bg-white/[0.04]",
              previewUrl && "border-solid border-medical-500/30 p-6"
            )}
            onClick={() => fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            whileHover={!previewUrl ? { scale: 1.02 } : {}}
            whileTap={!previewUrl ? { scale: 0.98 } : {}}
          >
            {/* Ambient glow behind dropzone */}
            <div className="absolute inset-0 bg-gradient-to-b from-medical-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />

            {previewUrl ? (
              <div className="flex flex-col items-center gap-6 relative z-10">
                <div className="relative rounded-2xl overflow-hidden border border-white/10 shadow-2xl bg-black/50 p-2">
                  <img src={previewUrl} alt="X-ray preview" className="max-h-72 rounded-xl object-contain opacity-90" />
                  <div className="absolute inset-0 ring-1 ring-inset ring-white/10 rounded-2xl pointer-events-none" />
                </div>
                <div className="flex items-center gap-2 text-sm font-medium text-zinc-300 bg-white/5 px-4 py-2 rounded-full border border-white/10 backdrop-blur-md">
                  <FileImage className="w-4 h-4 text-medical-400" />
                  {file?.name}
                  <span className="text-zinc-500 mx-2">|</span>
                  <span className="text-zinc-400 font-normal hover:text-medical-400 transition-colors">Change file</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-5 py-12 relative z-10">
                <div className="w-20 h-20 rounded-2xl bg-medical-500/10 text-medical-400 flex items-center justify-center mb-2 shadow-[0_0_20px_rgba(20,184,166,0.1)] group-hover:scale-110 transition-transform duration-300">
                  <UploadCloud className="w-10 h-10" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-zinc-100 mb-2">Select Diagnostic Image</h3>
                  <p className="text-sm text-zinc-400">Drag and drop your DICOM/JPEG file here, or click to browse</p>
                </div>
                <div className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold mt-2 px-3 py-1 rounded-full border border-white/5 bg-white/5">
                  Supported: JPEG, PNG, WEBP
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>

        {/* EHR Accordion Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="w-full max-w-2xl"
        >
          <div className="glass-card rounded-2xl overflow-hidden border border-white/10">
            <div 
              className="flex items-center justify-between p-5 cursor-pointer hover:bg-white/5 transition-colors"
              onClick={() => setShowEHR(!showEHR)}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center border border-white/5">
                  <Activity className="w-4 h-4 text-medical-400" />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-zinc-200">Patient Clinical Data</h4>
                  <p className="text-xs text-zinc-500">Provide optional EHR data to improve accuracy via multi-modal fusion.</p>
                </div>
              </div>
              <motion.div animate={{ rotate: showEHR ? 180 : 0 }} className="text-zinc-500">
                <ChevronDown className="w-5 h-5" />
              </motion.div>
            </div>

            <AnimatePresence>
              {showEHR && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: "easeInOut" }}
                  className="overflow-hidden border-t border-white/5 bg-black/20"
                >
                  <div className="p-6 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                    {EHR_FIELDS.map((field) => (
                      <div className="flex flex-col gap-2" key={field.key}>
                        <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-widest">{field.label}</label>
                        <input
                          type={field.type}
                          placeholder={field.placeholder}
                          value={ehr[field.key] || ""}
                          onChange={(e) =>
                            setEhr((prev) => ({ ...prev, [field.key]: e.target.value }))
                          }
                          className="w-full px-3 py-2 bg-zinc-900 border border-white/10 rounded-lg text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-medical-500 focus:border-medical-500 transition-all placeholder:text-zinc-700"
                        />
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Submit Button */}
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className={cn(
            "btn-primary w-full max-w-sm py-4 text-lg mt-2 group relative",
            (!file || loading) && "opacity-50 cursor-not-allowed grayscale"
          )}
          onClick={handleSubmit}
          disabled={!file || loading}
          whileHover={file && !loading ? { scale: 1.02 } : {}}
          whileTap={file && !loading ? { scale: 0.98 } : {}}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-medical-600 to-medical-400 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl" />
          <span className="relative z-10 flex items-center justify-center gap-3">
            {loading ? (
              <>
                <div className="spinner border-t-zinc-950/50" />
                Processing...
              </>
            ) : (
              <>
                <ScanFace className="w-5 h-5 group-hover:scale-110 transition-transform" /> 
                Run AI Diagnostics
              </>
            )}
          </span>
        </motion.button>

        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.6 }}
          className="text-xs text-zinc-600 text-center max-w-md mt-4 leading-relaxed font-mono"
        >
          FOR RESEARCH USE ONLY. NOT FOR CLINICAL DIAGNOSTIC DECISIONS.
        </motion.p>
      </main>
    </div>
  );
}
