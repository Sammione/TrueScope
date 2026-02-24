"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  ShieldCheck,
  BarChart3,
  Search,
  UploadCloud,
  Zap,
  MessageSquare,
  FileText,
  Activity,
  AlertTriangle,
  ChevronRight,
  RefreshCw,
  Plus,
  CheckCircle2,
  XCircle,
  Menu,
  X,
  Loader2,
  Leaf,
  Globe,
  TrendingDown
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// --- Utilities ---

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Components ---

const SidebarItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button
    onClick={onClick}
    className={cn(
      "w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all duration-300 group relative overflow-hidden",
      active
        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
        : "text-slate-500 hover:text-slate-200 hover:bg-white/5"
    )}
  >
    <div className={cn("absolute inset-0 bg-gradient-to-r from-emerald-500/0 via-emerald-500/5 to-emerald-500/0 translate-x-[-100%] transition-transform duration-1000", active && "group-hover:translate-x-[100%]")} />
    <Icon className={cn("w-5 h-5 transition-transform group-hover:scale-110", active && "text-emerald-400")} />
    <span className="text-sm font-medium font-outfit relative z-10">{label}</span>
    {active && (
      <motion.div
        layoutId="sidebar-accent"
        className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#10b981]"
      />
    )}
  </button>
);

const BentoCard = ({ children, title, className, icon: Icon, delay = 0, loading = false }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    className={cn("glass rounded-[2.5rem] p-6 lg:p-8 glass-hover flex flex-col gap-4 relative overflow-hidden", className)}
  >
    {loading && (
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm z-20 flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-500 animate-spin" />
      </div>
    )}
    <div className="flex items-center justify-between z-10">
      <div className="flex items-center gap-3">
        {Icon && (
          <div className="p-2.5 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 shadow-[0_0_15px_-5px_#10b981]">
            <Icon className="w-5 h-5" />
          </div>
        )}
        <h3 className="text-sm font-bold font-outfit uppercase tracking-widest text-slate-400">{title}</h3>
      </div>
    </div>
    <div className="flex-1 z-10 relative">
      {children}
    </div>
    {/* Ambient Glow */}
    <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-emerald-500/5 blur-[80px] rounded-full pointer-events-none" />
  </motion.div>
);

const RiskGauge = ({ score }: { score: string }) => {
  // Map score to percentage for the gauge
  const getPercentage = (s: string) => {
    switch (s?.toLowerCase()) {
      case "low": return 25;
      case "medium": return 50;
      case "high": return 85;
      default: return 0;
    }
  };

  const pct = getPercentage(score);
  const color = pct > 75 ? "#f43f5e" : pct > 40 ? "#fbbf24" : "#10b981"; // Red, Amber, Emerald
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="relative w-40 h-40 flex items-center justify-center">
      <svg className="w-full h-full -rotate-90">
        <circle cx="80" cy="80" r={radius} className="fill-none stroke-white/5 stroke-[12]" />
        <motion.circle
          cx="80" cy="80" r={radius}
          className="fill-none stroke-[12] drop-shadow-[0_0_10px_rgba(0,0,0,0.5)]"
          style={{ stroke: color }}
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-3xl font-black font-outfit"
          style={{ color }}
        >
          {score || "N/A"}
        </motion.span>
        <span className="text-[10px] uppercase tracking-widest font-bold text-slate-500">Risk Level</span>
      </div>
    </div>
  );
};

// --- Chat Component ---

const ChatView = ({ reportId }: { reportId: string | null }) => {
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant', content: string, citations?: any[] }[]>([
    { role: "assistant", content: "Hello! I've analyzed the report. Ask me anything about specific emissions, targets, or potential greenwashing risks." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || !reportId) return;
    const userMsg = input;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/api/query`, {
        question: userMsg,
        report_ids: [reportId]
      });
      setMessages(prev => [...prev, {
        role: "assistant",
        content: res.data.answer,
        citations: res.data.citations
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: "I encountered an error trying to answer that. Please check the backend connection." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[80vh] glass rounded-[2.5rem] overflow-hidden relative">
      <div className="flex-1 overflow-y-auto p-6 space-y-6" ref={scrollRef}>
        {messages.map((msg, i) => (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            key={i}
            className={cn("flex gap-4 max-w-3xl", msg.role === "user" ? "ml-auto flex-row-reverse" : "")}
          >
            <div className={cn(
              "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
              msg.role === "assistant" ? "bg-emerald-500 text-black" : "bg-white/10 text-white"
            )}>
              {msg.role === "assistant" ? <Zap className="w-4 h-4" /> : <div className="w-2 h-2 rounded-full bg-white" />}
            </div>
            <div className={cn(
              "p-4 rounded-2xl text-sm leading-relaxed",
              msg.role === "assistant" ? "bg-white/5 border border-white/5 text-slate-200" : "bg-emerald-500 text-black font-medium"
            )}>
              {msg.content}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/10 space-y-2">
                  <p className="text-[10px] uppercase tracking-widest font-bold opacity-50">Sources</p>
                  {msg.citations.map((c: any, ci: number) => (
                    <div key={ci} className="text-[10px] p-2 rounded bg-black/20 text-slate-400 font-mono">
                      Page {c.page}: "...{c.snippet.slice(0, 100)}..."
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        ))}
        {loading && (
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-emerald-500 text-black flex items-center justify-center shrink-0">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="p-4 rounded-2xl bg-white/5 border border-white/5 text-slate-400 text-sm italic">
              Analyzing report context...
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-white/5 bg-black/20 backdrop-blur-md">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask about specific claims, data, or contradictions..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/50 transition-all font-outfit"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-emerald-500 text-black rounded-xl hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

// --- ESG Radar Chart ---

const ESGRadarChart = ({ data }: { data: any }) => {
  const radarData = [
    { subject: 'Environmental', A: data.e || 20, fullMark: 100 },
    { subject: 'Social', A: data.s || 20, fullMark: 100 },
    { subject: 'Governance', A: data.g || 20, fullMark: 100 },
    { subject: 'Transparency', A: data.transparency || 20, fullMark: 100 },
    { subject: 'Verification', A: data.verification || 20, fullMark: 100 },
  ];

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700 }} />
          <Radar
            name="ESG Profile"
            dataKey="A"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.5}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

// --- Claims View Component ---

const ClaimsView = ({ claimsData }: { claimsData: any }) => {
  if (!claimsData) return <div className="text-center p-12 text-slate-500">No claims analysis available.</div>;

  return (
    <div className="flex flex-col gap-8">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: "Total Claims", value: claimsData.summary.total_claims, icon: FileText, color: "text-blue-400" },
          { label: "Supported", value: claimsData.summary.supported, icon: CheckCircle2, color: "text-emerald-400" },
          { label: "Weak/Vague", value: claimsData.summary.weak, icon: AlertTriangle, color: "text-amber-400" },
          { label: "Unsupported", value: (claimsData.summary.unsupported || 0) + (claimsData.summary.contradictory || 0), icon: XCircle, color: "text-rose-400" },
        ].map((stat, i) => (
          <div key={i} className="glass p-4 rounded-2xl border border-white/5 flex items-center gap-4">
            <div className={cn("p-2 rounded-xl bg-white/5", stat.color)}>
              <stat.icon className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">{stat.label}</p>
              <p className="text-xl font-black font-outfit">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4">
        {claimsData.results.map((item: any, i: number) => (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            key={i}
            className="glass p-6 rounded-[2rem] border border-white/5 space-y-4 hover:border-white/10 transition-all group"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className={cn(
                    "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
                    item.verdict === 'supported' ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" :
                      item.verdict === 'weak' ? "bg-amber-500/10 text-amber-400 border border-amber-500/20" :
                        "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                  )}>
                    {item.verdict}
                  </span>
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                    Confidence: {(item.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-lg font-bold font-outfit text-white leading-tight">
                  {item.claim.text}
                </p>
              </div>
              <div className="shrink-0 p-2 rounded-xl bg-white/5 text-slate-400 group-hover:text-white transition-colors">
                <ChevronRight className="w-5 h-5" />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/5">
              <div className="space-y-2">
                <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">Analysis Rationale</p>
                <p className="text-xs text-slate-300 leading-relaxed italic">"{item.rationale}"</p>
                {item.missing && (
                  <p className="text-[10px] text-slate-500"><span className="font-bold text-rose-400/70">Missing:</span> {item.missing}</p>
                )}
              </div>
              <div className="space-y-2">
                <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">Source Evidence</p>
                <div className="space-y-2">
                  {item.evidence.slice(0, 2).map((ev: any, ei: number) => (
                    <div key={ei} className="text-[10px] p-2 rounded-xl bg-black/40 border border-white/5 text-slate-400 font-mono line-clamp-2">
                      <span className="text-emerald-500 font-bold">[{ev.source}]</span> {ev.snippet}
                    </div>
                  ))}
                  {item.evidence.length === 0 && <p className="text-[10px] text-slate-600 italic">No direct evidence found in report.</p>}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// --- Reports View ---

const ReportsView = ({ allReports, onSelect }: { allReports: any[], onSelect: (reportId: string) => void }) => {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-black font-outfit tracking-tighter">Your Library</h2>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {allReports.map((report, i) => (
          <motion.div
            key={report.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="glass p-6 rounded-[2rem] border border-white/5 hover:border-emerald-500/30 transition-all cursor-pointer group"
          >
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 rounded-2xl bg-white/5 text-emerald-400 group-hover:bg-emerald-500/20 transition-colors">
                <FileText className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-white truncate max-w-[150px]">{report.name}</h4>
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                  {new Date(report.uploaded_at).toLocaleDateString()}
                </p>
              </div>
            </div>
            <p className="text-xs text-slate-400 mb-6">PDF Document • {report.pages} pages</p>
            <button
              onClick={() => onSelect(report.id)}
              className="w-full py-3 rounded-xl bg-white/5 text-xs font-bold hover:bg-emerald-500 hover:text-black transition-all"
            >
              Open Analysis
            </button>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// --- Metrics View ---

const MetricsView = ({ metrics, risk }: { metrics: any, risk: any }) => {
  if (!metrics) return null;

  const sections = [
    {
      title: "Carbon Emissions", data: [
        { label: "Scope 1 (tCO2e)", value: metrics.emissions?.scope1_tco2e },
        { label: "Scope 2 (tCO2e)", value: metrics.emissions?.scope2_tco2e },
        { label: "Scope 3 (tCO2e)", value: metrics.emissions?.scope3_tco2e },
      ]
    },
    {
      title: "Resources & Energy", data: [
        { label: "Total Energy (MWh)", value: metrics.energy?.total_mwh },
        { label: "Water Withdrawals (m³)", value: metrics.water?.withdrawals_m3 },
        { label: "Non-Hazardous Waste (Tonnes)", value: metrics.waste?.total_tonnes },
      ]
    },
    {
      title: "Social & Governance", data: [
        { label: "Total Employees", value: metrics.social?.employees_total },
        { label: "Board Diversity (Female %)", value: metrics.governance?.board_female_pct },
      ]
    }
  ];

  return (
    <div className="flex flex-col gap-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {sections.map((section, idx) => (
          <BentoCard key={idx} title={section.title} className="col-span-1">
            <div className="space-y-4 pt-2">
              {section.data.map((item, i) => (
                <div key={i} className="flex flex-col gap-1 p-3 rounded-2xl bg-white/[0.02] border border-white/5">
                  <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">{item.label}</span>
                  <span className="text-xl font-black font-outfit text-white">
                    {item.value !== null && item.value !== undefined ? item.value.toLocaleString() : "Not Disclosed"}
                  </span>
                </div>
              ))}
            </div>
          </BentoCard>
        ))}
      </div>
    </div>
  );
};


// --- Main Page ---

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [file, setFile] = useState<File | null>(null);
  const [reportId, setReportId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [claimsData, setClaimsData] = useState<any>(null);
  const [allReports, setAllReports] = useState<any[]>([]);
  const [isVerifyingClaims, setIsVerifyingClaims] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchReports = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/reports`);
      setAllReports(res.data.reports);
    } catch (e) {
      console.error("Failed to fetch reports:", e);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    executeUpload(selectedFile);
  };

  const executeUpload = async (fileToUpload: File) => {
    setFile(fileToUpload);
    setIsUploading(true);
    setAnalysisData(null);

    const formData = new FormData();
    formData.append("files", fileToUpload);

    try {
      // 1. Upload Report
      const uploadRes = await axios.post(`${API_URL}/api/reports`, formData);
      const reports = uploadRes.data.reports;
      // Get the ID of the last report in the list, assuming it's the one we just uploaded
      const newReportId = reports[reports.length - 1].id;
      setReportId(newReportId);
      setIsUploading(false);
      setIsAnalyzing(true);

      // 2. Run Analysis in Parallel
      const [riskRes, metricsRes, complianceRes] = await Promise.all([
        axios.post(`${API_URL}/api/risk`, { report_id: newReportId }),
        axios.post(`${API_URL}/api/metrics`, { report_id: newReportId }),
        axios.post(`${API_URL}/api/compliance`, { report_id: newReportId })
      ]);

      setAnalysisData({
        risk: riskRes.data,
        metrics: metricsRes.data.metrics,
        compliance: complianceRes.data.compliance
      });

      // 3. Extract and Verify Claims
      setIsVerifyingClaims(true);
      const claimsExtract = await axios.post(`${API_URL}/api/claims/extract`, { report_id: newReportId, max_claims: 12 });
      const claimsVerify = await axios.post(`${API_URL}/api/claims/verify`, {
        report_id: newReportId,
        claims: claimsExtract.data.claims,
        include_external_evidence: true
      });
      setClaimsData(claimsVerify.data);
      await fetchReports();
      setIsVerifyingClaims(false);

    } catch (error) {
      console.error("Error analyzing report:", error);
      alert("Analysis failed. Please check the backend connection.");
      setFile(null); // Reset
    } finally {
      setIsAnalyzing(false);
      setIsVerifyingClaims(false);
      setActiveTab("dashboard");
    }
  };

  const loadSampleReport = async () => {
    setIsUploading(true);
    try {
      const res = await axios.post(`${API_URL}/api/sample-report`);
      const reports = res.data.reports;
      const newReportId = reports[reports.length - 1].id;
      setReportId(newReportId);
      setFile({ name: "Sample Report (Demo)" } as File);
      setIsUploading(false);
      setIsAnalyzing(true);

      const [riskRes, metricsRes, complianceRes] = await Promise.all([
        axios.post(`${API_URL}/api/risk`, { report_id: newReportId }),
        axios.post(`${API_URL}/api/metrics`, { report_id: newReportId }),
        axios.post(`${API_URL}/api/compliance`, { report_id: newReportId })
      ]);

      setAnalysisData({
        risk: riskRes.data,
        metrics: metricsRes.data.metrics,
        compliance: complianceRes.data.compliance
      });

      // Claims for sample
      setIsVerifyingClaims(true);
      const claimsExtract = await axios.post(`${API_URL}/api/claims/extract`, { report_id: newReportId, max_claims: 8 });
      const claimsVerify = await axios.post(`${API_URL}/api/claims/verify`, {
        report_id: newReportId,
        claims: claimsExtract.data.claims,
        include_external_evidence: false
      });
      setClaimsData(claimsVerify.data);
      await fetchReports();
      setIsVerifyingClaims(false);

    } catch (e) {
      console.error(e);
      alert("Failed to load sample report.");
    } finally {
      setIsAnalyzing(false);
      setIsVerifyingClaims(false);
    }
  };

  const triggerUpload = () => fileInputRef.current?.click();

  // Prepare chart data safely
  const metricsChartData = analysisData?.metrics?.emissions ? [
    { name: "Scope 1", value: analysisData.metrics.emissions.scope1_tco2e || 0 },
    { name: "Scope 2", value: analysisData.metrics.emissions.scope2_tco2e || 0 },
    { name: "Scope 3", value: analysisData.metrics.emissions.scope3_tco2e || 0 },
  ] : [];

  return (
    <div className="flex min-h-screen bg-[#050505] text-white selection:bg-emerald-500/30 overflow-hidden flex-col md:flex-row">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-emerald-900/20 via-[#050505] to-black pointer-events-none" />

      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 left-0 right-0 p-4 z-50 flex items-center justify-between bg-black/80 backdrop-blur-md border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-emerald-500 flex items-center justify-center">
            <ShieldCheck className="text-black w-5 h-5" />
          </div>
          <span className="font-bold font-outfit text-white">TrueScope</span>
        </div>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 text-white">
          {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Sidebar (Responsive) */}
      <aside className={cn(
        "fixed inset-y-0 left-0 w-80 bg-black/90 backdrop-blur-xl z-40 p-8 flex flex-col gap-8 transition-transform duration-300 border-r border-white/5 md:relative md:bg-black/20 md:backdrop-blur-xl md:translate-x-0",
        isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex items-center gap-3 px-2 mt-16 md:mt-0">
          <div className="w-10 h-10 rounded-2xl bg-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <ShieldCheck className="text-black w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-outfit leading-tight">TrueScope</h1>
            <p className="text-[10px] text-emerald-500 font-bold tracking-widest uppercase">AI Verifier</p>
          </div>
        </div>

        <nav className="space-y-2 flex-1">
          <SidebarItem
            icon={Activity}
            label="Dashboard"
            active={activeTab === "dashboard"}
            onClick={() => { setActiveTab("dashboard"); setIsMobileMenuOpen(false); }}
          />
          <SidebarItem
            icon={MessageSquare}
            label="Ask AI"
            active={activeTab === "chat"}
            onClick={() => { setActiveTab("chat"); setIsMobileMenuOpen(false); }}
          />
          <SidebarItem
            icon={Search}
            label="Deep Dive"
            active={activeTab === "claims"}
            onClick={() => { setActiveTab("claims"); setIsMobileMenuOpen(false); }}
          />
          <SidebarItem
            icon={BarChart3}
            label="Metrics & Data"
            active={activeTab === "metrics"}
            onClick={() => { setActiveTab("metrics"); setIsMobileMenuOpen(false); }}
          />
          <SidebarItem
            icon={FileText}
            label="Reports"
            active={activeTab === "reports"}
            onClick={() => { setActiveTab("reports"); setIsMobileMenuOpen(false); }}
          />
        </nav>

        <div className="glass rounded-3xl p-6 space-y-4 relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          <div className="flex items-center gap-2 relative z-10">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-bold font-outfit">Pro Analytics</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed font-medium relative z-10">
            Unlock deep auditing and external truth-mapping today.
          </p>
          <button className="w-full py-3 rounded-2xl bg-emerald-500 text-black text-xs font-black uppercase tracking-wider hover:bg-emerald-400 transition-all relative z-10 shadow-lg shadow-emerald-500/20">
            Upgrade Now
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 md:p-12 lg:p-16 flex flex-col gap-12 max-w-7xl mx-auto relative z-10 pt-24 md:pt-12 overflow-y-auto">
        <AnimatePresence mode="wait">
          {!analysisData && !isAnalyzing ? (
            // --- Hero / Upload View ---
            <motion.div
              key="upload"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="flex-1 flex flex-col items-center justify-center min-h-[60vh] text-center"
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="mb-8 space-y-4"
              >
                <h1 className="text-5xl md:text-7xl font-black font-outfit tracking-tighter bg-gradient-to-b from-white to-white/40 bg-clip-text text-transparent">
                  Detect Greenwashing<br />
                  <span className="text-emerald-500">Instantly.</span>
                </h1>
                <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
                  Upload your ESG report and let our AI analyze claims against global standards (GRI, SASB) to reveal the truth behind the numbers.
                </p>
              </motion.div>

              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={triggerUpload}
                className="w-full max-w-xl h-64 border-2 border-dashed border-white/10 rounded-[2.5rem] bg-white/[0.02] hover:bg-white/[0.04] hover:border-emerald-500/50 transition-all cursor-pointer flex flex-col items-center justify-center gap-4 group relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-2xl" />
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  className="hidden"
                  accept=".pdf,.txt"
                />
                <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-emerald-500/20 group-hover:text-emerald-400 transition-colors">
                  {isUploading ? (
                    <Loader2 className="w-8 h-8 animate-spin" />
                  ) : (
                    <UploadCloud className="w-8 h-8" />
                  )}
                </div>
                <div className="relative z-10">
                  <p className="font-bold text-lg">Click to Upload Report</p>
                  <p className="text-sm text-slate-500">Support for PDF & TXT</p>
                </div>
              </motion.div>

              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                onClick={loadSampleReport}
                className="mt-6 text-sm text-emerald-500 font-bold hover:text-emerald-400 transition-colors flex items-center gap-2 px-4 py-2 rounded-full hover:bg-emerald-500/10 cursor-pointer z-20"
              >
                <Zap className="w-4 h-4" />
                No file? Try with a Sample Report
              </motion.button>
            </motion.div>
          ) : isAnalyzing ? (
            // --- Loading State ---
            <motion.div
              key="analyzing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex flex-col items-center justify-center gap-8"
            >
              <div className="relative w-24 h-24">
                <div className="absolute inset-0 rounded-full border-4 border-white/10" />
                <div className="absolute inset-0 rounded-full border-4 border-t-emerald-500 animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Zap className="w-8 h-8 text-emerald-500 animate-pulse" />
                </div>
              </div>
              <div className="text-center space-y-2">
                <h2 className="text-2xl font-bold font-outfit">Analyzing Report Data...</h2>
                <p className="text-slate-400">Checking Scope 1, 2, 3 • Verifying GRI Alignment • Detecting Anomalies</p>
              </div>
            </motion.div>
          ) : activeTab === "reports" ? (
            <motion.div
              key="reports"
              variants={containerVariants}
              initial="hidden"
              animate="show"
              className="w-full"
            >
              <ReportsView
                allReports={allReports}
                onSelect={async (id) => {
                  setIsAnalyzing(true);
                  setActiveTab("dashboard");
                  const [riskRes, metricsRes, complianceRes] = await Promise.all([
                    axios.post(`${API_URL}/api/risk`, { report_id: id }),
                    axios.post(`${API_URL}/api/metrics`, { report_id: id }),
                    axios.post(`${API_URL}/api/compliance`, { report_id: id })
                  ]);
                  setAnalysisData({
                    risk: riskRes.data,
                    metrics: metricsRes.data.metrics,
                    compliance: complianceRes.data.compliance
                  });
                  setReportId(id);
                  setIsAnalyzing(false);
                }}
              />
            </motion.div>
          ) : activeTab === "metrics" ? (
            <motion.div
              key="metrics"
              variants={containerVariants}
              initial="hidden"
              animate="show"
              className="w-full"
            >
              <header className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] uppercase font-black tracking-[0.3em] text-emerald-500">
                      Data Breakdown
                    </span>
                  </div>
                  <h2 className="text-4xl md:text-5xl font-black font-outfit tracking-tighter">Metric Repository</h2>
                </div>
              </header>
              <MetricsView metrics={analysisData?.metrics} risk={analysisData?.risk} />
            </motion.div>
          ) : (
            // --- Dashboard / Chat View Swapper ---
            activeTab === "chat" ? (
              <motion.div
                key="chat"
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className="w-full"
              >
                <ChatView reportId={reportId} />
              </motion.div>
            ) : activeTab === "claims" ? (
              <motion.div
                key="claims"
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className="w-full"
              >
                <header className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-[10px] uppercase font-black tracking-[0.3em] text-emerald-500">
                        Evidence Deep Dive
                      </span>
                    </div>
                    <h2 className="text-4xl md:text-5xl font-black font-outfit tracking-tighter">Claim Verifier</h2>
                  </div>
                </header>
                <ClaimsView claimsData={claimsData} />
              </motion.div>
            ) : (
              <motion.div
                key="dashboard"
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className="flex flex-col gap-8 w-full"
              >
                {/* Header */}
                <header className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-[10px] uppercase font-black tracking-[0.3em] text-emerald-500">
                        Analysis Complete
                      </span>
                      <div className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-[10px] font-bold text-emerald-400">
                        {file?.name}
                      </div>
                    </div>
                    <h2 className="text-4xl md:text-5xl font-black font-outfit tracking-tighter">Executive Summary</h2>
                  </div>
                  <div className="flex items-center gap-4">
                    <button onClick={() => { setAnalysisData(null); setClaimsData(null); }} className="flex items-center gap-2 px-6 py-3.5 rounded-full glass border-white/10 text-slate-300 text-sm font-bold hover:bg-white/10 transition-all">
                      <RefreshCw className="w-4 h-4" />
                      New Analysis
                    </button>
                    <button onClick={() => window.print()} className="flex items-center gap-2 px-6 py-3.5 rounded-full bg-emerald-500 text-black text-sm font-bold hover:bg-emerald-400 transition-all shadow-lg shadow-emerald-500/20">
                      <FileText className="w-4 h-4" />
                      Export Report
                    </button>
                  </div>
                </header>

                {/* Bento Grid */}
                <section className="grid grid-cols-1 md:grid-cols-6 lg:grid-cols-4 gap-6">

                  {/* Risk Gauge */}
                  <BentoCard title="Greenwashing Risk" className="md:col-span-3 lg:col-span-1" icon={AlertTriangle} delay={0.1}>
                    <div className="flex flex-col items-center justify-center h-full gap-4">
                      <RiskGauge score={analysisData.risk.score} />
                      <p className="text-[11px] text-center text-slate-500 leading-relaxed font-medium px-4 line-clamp-3">
                        {analysisData.risk.explanation}
                      </p>
                    </div>
                  </BentoCard>

                  {/* ESG Radar Chart */}
                  <BentoCard title="ESG Integrity Profile" className="md:col-span-3 lg:col-span-1" icon={Globe} delay={0.2}>
                    <ESGRadarChart data={{
                      e: analysisData.metrics.emissions ? 80 : 30,
                      s: analysisData.metrics.social ? 70 : 40,
                      g: analysisData.metrics.governance ? 90 : 50,
                      transparency: analysisData.compliance.gri?.covered ? 85 : 40,
                      verification: claimsData?.summary?.supported > 0 ? 75 : 30
                    }} />
                  </BentoCard>

                  {/* Emissions Chart */}
                  <BentoCard title="Emissions Profile (tCO2e)" className="md:col-span-6 lg:col-span-2" icon={Leaf} delay={0.3}>
                    <div className="h-48 w-full mt-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={metricsChartData}>
                          <XAxis
                            dataKey="name"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700 }}
                            dy={10}
                          />
                          <Tooltip
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            contentStyle={{ backgroundColor: '#000', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                          />
                          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                            {metricsChartData.map((entry: any, index: number) => (
                              <Cell key={`cell-${index}`} fill={index === 0 ? '#34d399' : index === 1 ? '#10b981' : '#059669'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </BentoCard>

                  {/* AI Insights / Key Metrics */}
                  <BentoCard title="Key Metrics Extracted" className="md:col-span-6 lg:col-span-4" icon={Activity} delay={0.4}>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {[
                        { label: "Energy (MWh)", value: analysisData.metrics.energy?.total_mwh || "N/A" },
                        { label: "Water (m³)", value: analysisData.metrics.water?.withdrawals_m3 || "N/A" },
                        { label: "Waste (Tonnes)", value: analysisData.metrics.waste?.total_tonnes || "N/A" },
                        { label: "Board Diversity", value: analysisData.metrics.governance?.board_female_pct ? `${analysisData.metrics.governance.board_female_pct}%` : "N/A" },
                      ].map((metric, i) => (
                        <div key={i} className="bg-white/5 rounded-2xl p-4 border border-white/5 flex flex-col gap-2">
                          <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">{metric.label}</span>
                          <span className="text-xl md:text-2xl font-black font-outfit text-white">{metric.value ? metric.value.toLocaleString() : "N/A"}</span>
                        </div>
                      ))}
                    </div>
                  </BentoCard>

                </section>
              </motion.div>
            )
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
