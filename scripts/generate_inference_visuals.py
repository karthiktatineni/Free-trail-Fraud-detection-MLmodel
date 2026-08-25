"""
Generate Visual Images of Inference Outputs, Genuine vs Fraud Scorecards,
and Attack Evasion Graph Traces with Clean Spacing & Typography.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "visuals", "inference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# 1. GENUINE USER SCORECARD (visuals/inference/genuine_user_result.png)
# ==============================================================================
def create_genuine_card():
    fig, (ax_main, ax_signals) = plt.subplots(1, 2, figsize=(14, 7), dpi=300, 
                                             gridspec_kw={'width_ratios': [1.25, 1]})
    fig.patch.set_facecolor('#0b1120')
    
    # Left Panel: Score Card
    ax_main.set_facecolor('#0f172a')
    ax_main.axis('off')
    
    # Title & Subtitle
    ax_main.text(0.06, 0.94, "REAL-TIME RISK ASSESSMENT REPORT", fontsize=11, fontweight='bold', color='#94a3b8', va='top')
    ax_main.text(0.06, 0.88, "SCENARIO: GENUINE USER SIGNUP", fontsize=15, fontweight='bold', color='#ffffff', va='top')
    
    # Verdict Box (Green)
    v_box = patches.FancyBboxPatch((0.06, 0.64), 0.88, 0.18, boxstyle="round,pad=0.02", 
                                  fc="#064e3b", ec="#10b981", lw=2)
    ax_main.add_patch(v_box)
    ax_main.text(0.10, 0.77, "VERDICT: NEW / GENUINE", fontsize=14, fontweight='bold', color='#a7f3d0', va='top')
    ax_main.text(0.10, 0.69, "ACTION: ALLOW FULL TRIAL ACCESS  |  CONFIDENCE: 99.5%", fontsize=9.5, fontweight='bold', color='#6ee7b7', va='top')
    
    # Score Section
    ax_main.text(0.06, 0.58, "RISK SCORE", fontsize=10, fontweight='bold', color='#94a3b8', va='top')
    ax_main.text(0.06, 0.52, "0.5", fontsize=38, fontweight='bold', color='#10b981', va='top')
    ax_main.text(0.25, 0.49, "/ 100", fontsize=13, fontweight='bold', color='#64748b', va='top')
    ax_main.text(0.48, 0.52, "Decision Threshold: 6.0\nOperating Band: [0.0 - 3.3] Allow", fontsize=9.5, color='#cbd5e1', va='top', linespacing=1.4)
    
    # Details Box
    details_box = patches.FancyBboxPatch((0.06, 0.06), 0.88, 0.32, boxstyle="round,pad=0.02", 
                                         fc="#1e293b", ec="#334155", lw=1.5)
    ax_main.add_patch(details_box)
    ax_main.text(0.10, 0.33, "USER IDENTITY ATTRIBUTES AUDIT", fontsize=10.5, fontweight='bold', color='#38bdf8', va='top')
    ax_main.text(0.10, 0.27, "• User Name: David Smith (ID: u_genuine_01)", fontsize=9, color='#f1f5f9', va='top')
    ax_main.text(0.10, 0.21, "• Email: david.smith@gmail.com (Established Clean Domain)", fontsize=9, color='#f1f5f9', va='top')
    ax_main.text(0.10, 0.15, "• Network: 203.0.113.45 (London, GB) | /24 Subnet: Unique", fontsize=9, color='#f1f5f9', va='top')
    ax_main.text(0.10, 0.09, "• Card & Device: pm_barclays_unique | dev_macbook_unique", fontsize=9, color='#f1f5f9', va='top')
    
    # Right Panel: Signals Bar Chart
    ax_signals.set_facecolor('#0f172a')
    signals = ['area_freq', 'device_os_freq', 'name_similarity', 'is_free_email', 'signup_hour']
    contributions = [0.01, 0.03, 0.15, 0.18, 2.30]
    
    y_pos = np.arange(len(signals))
    bars = ax_signals.barh(y_pos, contributions, color='#10b981', height=0.55, edgecolor='#34d399')
    
    ax_signals.set_yticks(y_pos)
    ax_signals.set_yticklabels(signals, fontsize=9.5, color='#e2e8f0', fontweight='bold')
    ax_signals.set_xlabel('Risk Signal Weight Contribution', fontsize=10, color='#94a3b8', fontweight='bold')
    ax_signals.set_title('Top Signal Breakdown (Protective/Low Risk)', fontsize=12, color='#ffffff', fontweight='bold', pad=12)
    ax_signals.tick_params(colors='#94a3b8')
    ax_signals.spines['top'].set_visible(False)
    ax_signals.spines['right'].set_visible(False)
    ax_signals.spines['left'].set_color('#334155')
    ax_signals.spines['bottom'].set_color('#334155')
    ax_signals.grid(axis='x', color='#1e293b', linestyle='--', alpha=0.7)
    
    for bar, val in zip(bars, contributions):
        ax_signals.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                        f"+{val:.2f}", va='center', ha='left', color='#6ee7b7', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "genuine_user_result.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("Generated visuals/inference/genuine_user_result.png")

# ==============================================================================
# 2. FRAUD SYNDICATE SCORECARD (visuals/inference/fraud_syndicate_result.png)
# ==============================================================================
def create_fraud_card():
    fig, (ax_main, ax_signals) = plt.subplots(1, 2, figsize=(14, 7), dpi=300, 
                                             gridspec_kw={'width_ratios': [1.25, 1]})
    fig.patch.set_facecolor('#0b1120')
    
    # Left Panel: Score Card
    ax_main.set_facecolor('#0f172a')
    ax_main.axis('off')
    
    # Title & Subtitle
    ax_main.text(0.06, 0.94, "REAL-TIME RISK ASSESSMENT REPORT", fontsize=11, fontweight='bold', color='#94a3b8', va='top')
    ax_main.text(0.06, 0.88, "SCENARIO: FRAUD SYNDICATE ATTACK", fontsize=15, fontweight='bold', color='#ffffff', va='top')
    
    # Verdict Box (Red)
    v_box = patches.FancyBboxPatch((0.06, 0.64), 0.88, 0.18, boxstyle="round,pad=0.02", 
                                  fc="#7f1d1d", ec="#ef4444", lw=2)
    ax_main.add_patch(v_box)
    ax_main.text(0.10, 0.77, "VERDICT: REPEAT / LIKELY ABUSE", fontsize=14, fontweight='bold', color='#fecaca', va='top')
    ax_main.text(0.10, 0.69, "ACTION: BLOCK TRIAL / REQUIRE PAYMENT  |  CONFIDENCE: 99.9%", fontsize=9, fontweight='bold', color='#fca5a5', va='top')
    
    # Score Section
    ax_main.text(0.06, 0.58, "RISK SCORE", fontsize=10, fontweight='bold', color='#94a3b8', va='top')
    ax_main.text(0.06, 0.52, "99.9", fontsize=38, fontweight='bold', color='#ef4444', va='top')
    ax_main.text(0.33, 0.49, "/ 100", fontsize=13, fontweight='bold', color='#64748b', va='top')
    ax_main.text(0.53, 0.52, "Decision Threshold: 6.0\nOperating Band: [6.0 - 100.0] Hard Block", fontsize=9.5, color='#cbd5e1', va='top', linespacing=1.4)
    
    # Details Box
    details_box = patches.FancyBboxPatch((0.06, 0.06), 0.88, 0.32, boxstyle="round,pad=0.02", 
                                         fc="#1e293b", ec="#334155", lw=1.5)
    ax_main.add_patch(details_box)
    ax_main.text(0.10, 0.33, "ADVERSARIAL ATTRIBUTES AUDIT", fontsize=10.5, fontweight='bold', color='#f87171', va='top')
    ax_main.text(0.10, 0.27, "• User Name: Sanjay Nair 2 (ID: demo_fraud_syndicate_03)", fontsize=9, color='#f1f5f9', va='top')
    ax_main.text(0.10, 0.21, "• Email: sanjay.nair+trial4@mailinator.com (Disposable Domain + Tag)", fontsize=9, color='#f1f5f9', va='top')
    ax_main.text(0.10, 0.15, "• Network: 39.173.180.190 (/24 Subnet: 39.173.180 High Velocity)", fontsize=9, color='#f1f5f9', va='top')
    ax_main.text(0.10, 0.09, "• Card & Device: pm_424776171fe7 (Reused across 4 syndicates)", fontsize=9, color='#f1f5f9', va='top')
    
    # Right Panel: Signals Bar Chart
    ax_signals.set_facecolor('#0f172a')
    signals = ['name_similarity', 'email_has_digits', 'signup_hour (02:00)', 'is_disposable_domain', 'email_plus_tag']
    contributions = [0.14, 0.19, 0.33, 6.97, 18.68]
    
    y_pos = np.arange(len(signals))
    bars = ax_signals.barh(y_pos, contributions, color='#ef4444', height=0.55, edgecolor='#f87171')
    
    ax_signals.set_yticks(y_pos)
    ax_signals.set_yticklabels(signals, fontsize=9.5, color='#e2e8f0', fontweight='bold')
    ax_signals.set_xlabel('Risk Signal Weight Contribution', fontsize=10, color='#94a3b8', fontweight='bold')
    ax_signals.set_title('Top Contributing Abuse Risk Signals', fontsize=12, color='#ffffff', fontweight='bold', pad=12)
    ax_signals.tick_params(colors='#94a3b8')
    ax_signals.spines['top'].set_visible(False)
    ax_signals.spines['right'].set_visible(False)
    ax_signals.spines['left'].set_color('#334155')
    ax_signals.spines['bottom'].set_color('#334155')
    ax_signals.grid(axis='x', color='#1e293b', linestyle='--', alpha=0.7)
    
    for bar, val in zip(bars, contributions):
        ax_signals.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                        f"+{val:.2f}", va='center', ha='left', color='#fca5a5', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fraud_syndicate_result.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("Generated visuals/inference/fraud_syndicate_result.png")

# ==============================================================================
# 3. ATTACK EVASION GRAPH TRACE (visuals/inference/attack_evasion_graph_trace.png)
# ==============================================================================
def create_evasion_trace():
    fig, (ax_trace, ax_metrics) = plt.subplots(1, 2, figsize=(15, 7), dpi=300, 
                                               gridspec_kw={'width_ratios': [1.15, 1]})
    fig.patch.set_facecolor('#0b1120')
    
    # Left Panel: Step-by-Step Evasion Diagram
    ax_trace.set_facecolor('#0f172a')
    ax_trace.axis('off')
    
    ax_trace.text(0.05, 0.95, "ADVERSARIAL ROTATION EVASION TRACE", fontsize=13, fontweight='bold', color='#ffffff', va='top')
    ax_trace.text(0.05, 0.89, "Incremental Union-Find Graph Neutralizing Attribute Rotation", fontsize=9.5, color='#94a3b8', va='top')
    
    # Step 1 Box
    s1_box = patches.FancyBboxPatch((0.05, 0.62), 0.90, 0.22, boxstyle="round,pad=0.02", fc="#1e293b", ec="#3b82f6", lw=1.5)
    ax_trace.add_patch(s1_box)
    ax_trace.text(0.08, 0.81, "ATTEMPT #1 : Initial Signup (Organic Heuristics)", fontsize=10, fontweight='bold', color='#60a5fa', va='top')
    ax_trace.text(0.08, 0.75, "• Email: vikram.patel@gmail.com  |  IP: 45.33.32.10  |  Card: pm_ring_card_99", fontsize=8.5, color='#cbd5e1', va='top')
    ax_trace.text(0.08, 0.69, "• State: Graph Size = 1 node  |  Subnet 24h Velocity = 0", fontsize=8.5, color='#94a3b8', va='top')
    ax_trace.text(0.08, 0.64, "► RISK SCORE: 13.0 / 100  (Flagged by odd-hour + lexical cluster)", fontsize=9, fontweight='bold', color='#fbbf24', va='top')
    
    # Step 2 Box
    s2_box = patches.FancyBboxPatch((0.05, 0.34), 0.90, 0.22, boxstyle="round,pad=0.02", fc="#1e293b", ec="#f59e0b", lw=1.5)
    ax_trace.add_patch(s2_box)
    ax_trace.text(0.08, 0.53, "ATTEMPT #2 : Rotated Email & IP (Same /24 Subnet + Card)", fontsize=10, fontweight='bold', color='#fbbf24', va='top')
    ax_trace.text(0.08, 0.47, "• Email: vikram.p+trial2@mailinator.com (ROTATED)  |  IP: 45.33.32.55 (ROTATED)", fontsize=8.5, color='#cbd5e1', va='top')
    ax_trace.text(0.08, 0.41, "• State: Graph Size = 3 nodes (Linked to Attempt #1) | Subnet Velocity = 1", fontsize=8.5, color='#94a3b8', va='top')
    ax_trace.text(0.08, 0.36, "► RISK SCORE: 100.0 / 100  (VERDICT: REPEAT ABUSE -> HARD BLOCK)", fontsize=9, fontweight='bold', color='#ef4444', va='top')
    
    # Step 3 Box
    s3_box = patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.22, boxstyle="round,pad=0.02", fc="#1e293b", ec="#ef4444", lw=2)
    ax_trace.add_patch(s3_box)
    ax_trace.text(0.08, 0.25, "ATTEMPT #3 : Rotated Device Fingerprint + New Disposable", fontsize=10, fontweight='bold', color='#f87171', va='top')
    ax_trace.text(0.08, 0.19, "• Email: vpatel99@tempmail.com (ROTATED)  |  Device: dev_new_phone (ROTATED)", fontsize=8.5, color='#cbd5e1', va='top')
    ax_trace.text(0.08, 0.13, "• State: Graph Component = 3 connected entities | Card Reuse = 2", fontsize=8.5, color='#94a3b8', va='top')
    ax_trace.text(0.08, 0.08, "► RISK SCORE: 100.0 / 100  (LOCKED & PERMANENTLY BLOCKED)", fontsize=9, fontweight='bold', color='#ef4444', va='top')
    
    # Right Panel: Velocity & Score Trajectory
    ax_metrics.set_facecolor('#0f172a')
    attempts = [1, 2, 3]
    scores = [13.0, 100.0, 100.0]
    graph_sizes = [1, 3, 3]
    
    ax_metrics.plot(attempts, scores, marker='o', markersize=8, color='#ef4444', lw=3, label='Risk Score (0-100)')
    ax_metrics.axhline(6.0, color='#f59e0b', linestyle='--', lw=1.5, label='Decision Threshold (T=6.0)')
    
    ax_metrics.set_xticks(attempts)
    ax_metrics.set_xticklabels(['Attempt #1\n(Seed)', 'Attempt #2\n(Rotated Email/IP)', 'Attempt #3\n(Rotated Device)'], 
                               fontsize=9, color='#e2e8f0', fontweight='bold')
    ax_metrics.set_ylabel('Model Risk Score', fontsize=10, color='#ef4444', fontweight='bold')
    ax_metrics.set_ylim(-5, 110)
    ax_metrics.set_title('Risk Score & Graph Cluster Size Trajectory', fontsize=12, color='#ffffff', fontweight='bold', pad=12)
    ax_metrics.tick_params(colors='#94a3b8')
    ax_metrics.grid(color='#1e293b', linestyle='--', alpha=0.7)
    
    # Twin axis for graph size
    ax_twin = ax_metrics.twinx()
    ax_twin.plot(attempts, graph_sizes, marker='s', markersize=8, color='#38bdf8', lw=2.5, linestyle=':', label='Connected Entities')
    ax_twin.set_ylabel('Graph Component Size (Entities)', fontsize=10, color='#38bdf8', fontweight='bold')
    ax_twin.set_ylim(0, 5)
    ax_twin.tick_params(colors='#38bdf8')
    
    # Combined legend
    lines1, labels1 = ax_metrics.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_metrics.legend(lines1 + lines2, labels1 + labels2, loc='center right', facecolor='#1e293b', edgecolor='#334155', fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attack_evasion_graph_trace.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("Generated visuals/inference/attack_evasion_graph_trace.png")

# ==============================================================================
# 4. REDESIGNED CLEAN LIVE DASHBOARD SUMMARY (visuals/inference/live_scoring_dashboard_summary.png)
# ==============================================================================
def create_dashboard_summary():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    fig.patch.set_facecolor('#0b1120')
    ax.set_facecolor('#0f172a')
    ax.axis('off')
    
    # Header
    ax.text(0.04, 0.96, "FRAUDGUARD AI: 3-BAND POLICY & LIVE SCORING ENGINE", fontsize=15, fontweight='bold', color='#ffffff', va='top')
    ax.text(0.04, 0.91, "Operational Risk Decisioning Framework Across Customer Population", fontsize=10, color='#94a3b8', va='top')
    
    # ------------------ BAND 1: GENUINE ------------------
    b1 = patches.FancyBboxPatch((0.04, 0.44), 0.29, 0.42, boxstyle="round,pad=0.02", fc="#064e3b", ec="#10b981", lw=2)
    ax.add_patch(b1)
    
    # Pill Header
    b1_pill = patches.FancyBboxPatch((0.06, 0.79), 0.25, 0.05, boxstyle="round,pad=0.01", fc="#047857", ec="#34d399", lw=1)
    ax.add_patch(b1_pill)
    ax.text(0.185, 0.815, "BAND 1: GENUINE", fontsize=10.5, fontweight='bold', color='#ffffff', ha='center', va='center')
    
    ax.text(0.06, 0.74, "Score Range: 0.0 to 3.3", fontsize=9.5, fontweight='bold', color='#a7f3d0', va='top')
    ax.text(0.06, 0.68, "• Action: ALLOW Instant Access", fontsize=9, color='#ffffff', fontweight='bold', va='top')
    ax.text(0.06, 0.63, "• Friction: Zero Friction", fontsize=8.5, color='#cbd5e1', va='top')
    ax.text(0.06, 0.58, "• Persona: Organic single human signups", fontsize=8.5, color='#cbd5e1', va='top')
    ax.text(0.06, 0.53, "• Volume: ~70.1% of all traffic", fontsize=8.5, color='#cbd5e1', va='top')
    
    ax.text(0.06, 0.47, "Example: David Smith (Score: 0.5/100)", fontsize=8.5, color='#6ee7b7', fontweight='bold', va='top')
    
    # ------------------ BAND 2: GREY ZONE ------------------
    b2 = patches.FancyBboxPatch((0.355, 0.44), 0.29, 0.42, boxstyle="round,pad=0.02", fc="#78350f", ec="#f59e0b", lw=2)
    ax.add_patch(b2)
    
    # Pill Header
    b2_pill = patches.FancyBboxPatch((0.375, 0.79), 0.25, 0.05, boxstyle="round,pad=0.01", fc="#b45309", ec="#fbbf24", lw=1)
    ax.add_patch(b2_pill)
    ax.text(0.50, 0.815, "BAND 2: GREY ZONE", fontsize=10.5, fontweight='bold', color='#ffffff', ha='center', va='center')
    
    ax.text(0.375, 0.74, "Score Range: 3.3 to 6.0", fontsize=9.5, fontweight='bold', color='#fde68a', va='top')
    ax.text(0.375, 0.68, "• Action: STEP-UP VERIFICATION", fontsize=9, color='#ffffff', fontweight='bold', va='top')
    ax.text(0.375, 0.63, "• Friction: SMS OTP / CAPTCHA / $1 Auth", fontsize=8.5, color='#cbd5e1', va='top')
    ax.text(0.375, 0.58, "• Persona: Foreign travel / Shared NAT", fontsize=8.5, color='#cbd5e1', va='top')
    ax.text(0.375, 0.53, "• Volume: ~4.5% of edge-case traffic", fontsize=8.5, color='#cbd5e1', va='top')
    
    ax.text(0.375, 0.47, "Example: Alex Johnson (Score: 4.2/100)", fontsize=8.5, color='#fbbf24', fontweight='bold', va='top')
    
    # ------------------ BAND 3: REPEAT ABUSE ------------------
    b3 = patches.FancyBboxPatch((0.67, 0.44), 0.29, 0.42, boxstyle="round,pad=0.02", fc="#7f1d1d", ec="#ef4444", lw=2)
    ax.add_patch(b3)
    
    # Pill Header
    b3_pill = patches.FancyBboxPatch((0.69, 0.79), 0.25, 0.05, boxstyle="round,pad=0.01", fc="#b91c1c", ec="#f87171", lw=1)
    ax.add_patch(b3_pill)
    ax.text(0.815, 0.815, "BAND 3: REPEAT ABUSE", fontsize=10.5, fontweight='bold', color='#ffffff', ha='center', va='center')
    
    ax.text(0.69, 0.74, "Score Range: 6.0 to 100.0", fontsize=9.5, fontweight='bold', color='#fecaca', va='top')
    ax.text(0.69, 0.68, "• Action: HARD BLOCK TRIAL ACCESS", fontsize=9, color='#ffffff', fontweight='bold', va='top')
    ax.text(0.69, 0.63, "• Friction: Require Paid Plan Upfront", fontsize=8.5, color='#cbd5e1', va='top')
    ax.text(0.69, 0.58, "• Persona: Multi-accounting syndicates", fontsize=8.5, color='#cbd5e1', va='top')
    ax.text(0.69, 0.53, "• Volume: ~29.9% of abusive traffic", fontsize=8.5, color='#cbd5e1', va='top')
    
    ax.text(0.69, 0.47, "Example: Sanjay Nair (Score: 99.9/100)", fontsize=8.5, color='#fca5a5', fontweight='bold', va='top')
    
    # ------------------ BOTTOM ARCHITECTURE SUMMARY ------------------
    summary_box = patches.FancyBboxPatch((0.04, 0.05), 0.92, 0.33, boxstyle="round,pad=0.02", fc="#1e293b", ec="#334155", lw=1.5)
    ax.add_patch(summary_box)
    ax.text(0.07, 0.33, "CORE ARCHITECTURAL PILLARS & PRODUCTION BENCHMARKS", fontsize=11, fontweight='bold', color='#38bdf8', va='top')
    ax.text(0.07, 0.26, "• Incremental Union-Find Graph: Disjoint-set linkage captures syndicate clusters (r=0.746), surviving email/IP rotation.", fontsize=9, color='#e2e8f0', va='top')
    ax.text(0.07, 0.19, "• 10-Fold CV Model Selection: XGBoost selected across 7 models (0.9305 CV Recall, 0.9621 F1-Score, 0.9725 ROC-AUC).", fontsize=9, color='#e2e8f0', va='top')
    ax.text(0.07, 0.12, "• Precision SLA Threshold Tuning: Optimal T=0.060 achieves 95.1% Abuse Recall with 78.8% Precision in <20ms latency.", fontsize=9, color='#e2e8f0', va='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "live_scoring_dashboard_summary.png"), dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print("Generated visuals/inference/live_scoring_dashboard_summary.png (Cleaned & Redesigned)")

if __name__ == "__main__":
    create_genuine_card()
    create_fraud_card()
    create_evasion_trace()
    create_dashboard_summary()
    print("\nAll 4 inference output visuals regenerated successfully in visuals/inference/")
