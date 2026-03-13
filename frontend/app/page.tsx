"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
    ShieldCheck,
    Search,
    BarChart3,
    FileText,
    ArrowRight,
    CheckCircle2,
    Zap,
    Globe
} from "lucide-react";

export default function LandingPage() {
    const containerVariants = {
        hidden: { opacity: 0 },
        show: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
            },
        },
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
    };

    return (
        <div className="min-h-screen bg-[#050505] text-white overflow-hidden font-outfit selection:bg-emerald-500/30">
            {/* Background Gradients */}
            <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-emerald-900/20 via-[#050505] to-black pointer-events-none" />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-emerald-500/10 blur-[120px] rounded-full pointer-events-none" />

            {/* Navigation */}
            <nav className="relative z-50 flex items-center justify-between px-6 py-6 max-w-7xl mx-auto">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                        <ShieldCheck className="text-black w-6 h-6" />
                    </div>
                    <span className="text-xl font-black tracking-tight">TrueScope</span>
                </div>
                <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300">
                    <a href="#features" className="hover:text-white transition-colors">Features</a>
                    <a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a>
                    <a href="#compliance" className="hover:text-white transition-colors">Compliance</a>
                </div>
                <Link href="/dashboard" className="px-5 py-2.5 rounded-full bg-white/5 border border-white/10 text-sm font-bold hover:bg-emerald-500 hover:text-black hover:border-emerald-500 transition-all shadow-lg hover:shadow-emerald-500/20">
                    Launch App
                </Link>
            </nav>

            <main className="relative z-10 flex flex-col items-center">
                {/* Hero Section */}
                <section className="w-full max-w-7xl mx-auto px-6 pt-24 pb-32 flex flex-col items-center text-center">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.7, ease: "easeOut" }}
                        className="flex items-center gap-2 px-4 py-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-bold uppercase tracking-widest mb-8"
                    >
                        <span className="relative flex h-2 w-2 mr-1">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Live News Fact-Checking Now Active
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1, duration: 0.5 }}
                        className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tighter leading-[1.1] mb-8"
                    >
                        Stop Guessing.<br />
                        <span className="bg-gradient-to-r from-emerald-400 to-teal-200 bg-clip-text text-transparent">Start Investigating.</span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2, duration: 0.5 }}
                        className="text-lg md:text-xl text-slate-400 max-w-2xl leading-relaxed mb-12"
                    >
                        TrueScope is the AI-powered due diligence engine that instantly detects greenwashing, cross-references corporate ESG claims against real-time news, and generates 1-click audit reports.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, duration: 0.5 }}
                        className="flex flex-col sm:flex-row items-center gap-4 w-full justify-center"
                    >
                        <Link href="/dashboard" className="w-full sm:w-auto px-8 py-4 rounded-full bg-emerald-500 text-black text-sm font-black uppercase tracking-wider hover:bg-emerald-400 hover:scale-105 transition-all shadow-[0_0_30px_-5px_#10b981] flex items-center justify-center gap-2 group">
                            Start Free Trial
                            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </Link>
                        <Link href="#demo" className="w-full sm:w-auto px-8 py-4 rounded-full bg-white/5 border border-white/10 text-white text-sm font-bold uppercase tracking-wider hover:bg-white/10 transition-all flex items-center justify-center">
                            Watch Demo
                        </Link>
                    </motion.div>
                </section>

                {/* The Differentiators Section */}
                <section id="features" className="w-full py-24 bg-black/40 border-y border-white/5">
                    <div className="max-w-7xl mx-auto px-6">
                        <div className="text-center mb-16">
                            <h2 className="text-3xl md:text-4xl font-black tracking-tight mb-4">Investigative Superpowers</h2>
                            <p className="text-slate-400">Why top-tier compliance officers and investors trust TrueScope.</p>
                        </div>

                        <motion.div
                            variants={containerVariants}
                            initial="hidden"
                            whileInView="show"
                            viewport={{ once: true, margin: "-100px" }}
                            className="grid grid-cols-1 md:grid-cols-3 gap-8"
                        >
                            {[
                                {
                                    icon: Search,
                                    title: "Live News Fact-Checker",
                                    description: "We don't just read the PDF. The moment you upload, TrueScope searches the live web for recent environmental lawsuits and NGOs contradicting their claims.",
                                    color: "emerald"
                                },
                                {
                                    icon: BarChart3,
                                    title: "Fluff-to-Fact Gauge",
                                    description: "Instantly algorithmically determine if a report is 80% flowery marketing speak or hard, verifiable tCO2e data.",
                                    color: "teal"
                                },
                                {
                                    icon: FileText,
                                    title: "1-Click CSRD Audits",
                                    description: "Download a comprehensive, structured Markdown document summarizing missing metrics, vague claims, and verified facts ready for investor review.",
                                    color: "indigo"
                                }
                            ].map((feature, i) => (
                                <motion.div key={i} variants={itemVariants} className="glass p-8 rounded-[2rem] border border-white/5 hover:border-white/10 transition-all group relative overflow-hidden">
                                    <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-duration-500" />
                                    <div className={"w-14 h-14 rounded-2xl flex items-center justify-center mb-6 " + (feature.color === "emerald" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : feature.color === "teal" ? "bg-teal-500/10 text-teal-400 border border-teal-500/20" : "bg-indigo-500/10 text-indigo-400 border border-indigo-500/20")}>
                                        <feature.icon className="w-7 h-7" />
                                    </div>
                                    <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                                    <p className="text-sm text-slate-400 leading-relaxed">{feature.description}</p>
                                </motion.div>
                            ))}
                        </motion.div>
                    </div>
                </section>

                {/* How it works (Interactive Demo placeholder) */}
                <section id="how-it-works" className="w-full py-32 max-w-7xl mx-auto px-6">
                    <div className="flex flex-col lg:flex-row items-center gap-16">
                        <div className="flex-1 space-y-8">
                            <h2 className="text-4xl md:text-5xl font-black tracking-tight leading-tight">
                                Turn a 200-page PDF into actionable truth in 40 seconds.
                            </h2>
                            <div className="space-y-6">
                                {[
                                    { step: "01", text: "Drag & drop corporate sustainability reports into the engine." },
                                    { step: "02", text: "TrueScope's RAG architecture parses and stores the document without hallucination." },
                                    { step: "03", text: "We extract core metrics, verify claims against global standards, and check the live web." },
                                ].map((item, i) => (
                                    <div key={i} className="flex gap-4 items-start">
                                        <div className="text-emerald-500 font-black font-mono mt-1">{item.step}</div>
                                        <div className="text-lg text-slate-300">{item.text}</div>
                                    </div>
                                ))}
                            </div>
                            <Link href="/dashboard" className="inline-flex items-center gap-2 text-emerald-400 font-bold hover:text-emerald-300 transition-colors">
                                Try it out now <ArrowRight className="w-4 h-4" />
                            </Link>
                        </div>
                        <div className="flex-1 w-full relative">
                            <div className="absolute -inset-4 bg-gradient-to-tr from-emerald-500/20 to-teal-500/20 blur-3xl opacity-50 rounded-full" />
                            <div className="glass rounded-[2rem] border border-white/10 p-6 relative overflow-hidden shadow-2xl">
                                {/* Mock UI elements */}
                                <div className="flex items-center justify-between mb-8 pb-4 border-b border-white/5">
                                    <div className="flex items-center gap-3">
                                        <div className="w-3 h-3 rounded-full bg-rose-500" />
                                        <div className="w-3 h-3 rounded-full bg-amber-500" />
                                        <div className="w-3 h-3 rounded-full bg-emerald-500" />
                                    </div>
                                    <span className="text-[10px] uppercase font-bold text-slate-500">TrueScope Dashboard Preview</span>
                                </div>
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5">
                                        <div className="flex items-center gap-3">
                                            <Zap className="w-5 h-5 text-emerald-400" />
                                            <span className="text-sm font-bold">Greenwashing Risk Score</span>
                                        </div>
                                        <span className="text-emerald-400 font-black">Medium-Low</span>
                                    </div>
                                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5">
                                        <div className="flex items-center gap-3">
                                            <BarChart3 className="w-5 h-5 text-amber-400" />
                                            <span className="text-sm font-bold">Fluff-to-Fact Ratio</span>
                                        </div>
                                        <span className="text-amber-400 font-black">65% Fluff / 35% Fact</span>
                                    </div>
                                    <div className="p-4 rounded-xl bg-black/40 border border-rose-500/20">
                                        <div className="flex items-start gap-3">
                                            <Globe className="w-5 h-5 text-rose-400 mt-1 shrink-0" />
                                            <div>
                                                <span className="text-xs font-bold text-rose-400 uppercase tracking-widest block mb-1">Contradiction Detected</span>
                                                <p className="text-sm text-slate-300">Live news indicates a pending environmental fine contradicting Section 4.1 claims.</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Global Standards Banner */}
                <section id="compliance" className="w-full py-16 bg-emerald-950/20 border-y border-emerald-900/30 text-center">
                    <p className="text-xs text-emerald-500 font-bold uppercase tracking-widest mb-6">Cross-references against major global frameworks</p>
                    <div className="flex flex-wrap items-center justify-center gap-8 md:gap-16 opacity-50 grayscale hover:grayscale-0 transition-all duration-500">
                        {["GRI", "SASB", "TCFD", "IFRS S1 & S2", "CSRD"].map((framework, i) => (
                            <span key={i} className="text-xl md:text-3xl font-black tracking-tighter text-white">{framework}</span>
                        ))}
                    </div>
                </section>

                {/* Final CTA */}
                <section className="w-full py-32 max-w-4xl mx-auto px-6 text-center">
                    <h2 className="text-4xl md:text-6xl font-black tracking-tight mb-6">Ready to find the truth?</h2>
                    <p className="text-lg text-slate-400 mb-10 max-w-2xl mx-auto">Join leading compliance teams and investors protecting their capital from fraudulent sustainability claims.</p>
                    <Link href="/dashboard" className="px-10 py-5 rounded-full bg-emerald-500 text-black text-lg font-black uppercase tracking-wider hover:bg-emerald-400 hover:scale-105 transition-all shadow-[0_0_30px_-5px_#10b981] inline-flex items-center justify-center gap-3 group">
                        <ShieldCheck className="w-6 h-6" />
                        Launch Dashboard Now
                    </Link>
                </section>
            </main>

            {/* Footer */}
            <footer className="w-full py-8 border-t border-white/5 text-center text-slate-600 text-sm">
                <p>&copy; 2026 TrueScope AI. All rights reserved.</p>
            </footer>
        </div>
    );
}
