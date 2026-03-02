/**
 * Insurance AI Assistant - Main Frontend Application
 * ====================================================
 * Handles navigation, API calls, chart rendering, and all UI interactions.
 * Uses Chart.js for visualizations.
 */

const API = '';  // Same origin — FastAPI serves both API and frontend

// -------------------------------------------------------------------
// Navigation
// -------------------------------------------------------------------
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const page = item.dataset.page;
        // Update sidebar
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        // Update pages
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(`page-${page}`).classList.add('active');
        // Lazy-load page data
        if (page === 'analytics') loadAnalytics();
        if (page === 'fraud') loadFeatureImportances();
    });
});

// -------------------------------------------------------------------
// Utility helpers
// -------------------------------------------------------------------
async function apiCall(url, options = {}) {
    try {
        const res = await fetch(API + url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API error [${url}]:`, err);
        return null;
    }
}

function $(id) { return document.getElementById(id); }

function formatMoney(n) {
    if (n == null) return '$0';
    return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function formatPct(n) { return (n != null ? n.toFixed(1) : '0') + '%'; }

// Chart instance cache (destroy before re-creating)
const charts = {};
function makeChart(canvasId, config) {
    if (charts[canvasId]) charts[canvasId].destroy();
    const ctx = $(canvasId)?.getContext('2d');
    if (!ctx) return null;
    charts[canvasId] = new Chart(ctx, config);
    return charts[canvasId];
}

// -------------------------------------------------------------------
// DASHBOARD - Load on startup
// -------------------------------------------------------------------
async function loadDashboard() {
    // Summary stats
    const summary = await apiCall('/api/analytics/summary');
    if (summary?.data) {
        const d = summary.data;
        $('dashboard-stats').innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${d.total_policies.toLocaleString()}</div>
                <div class="stat-label">Total Policies</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">${d.total_claims.toLocaleString()}</div>
                <div class="stat-label">Total Claims</div>
            </div>
            <div class="stat-card danger">
                <div class="stat-value">${d.fraud_count}</div>
                <div class="stat-label">Fraudulent Claims (${d.fraud_rate_pct}%)</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">${formatMoney(d.avg_claim_amount)}</div>
                <div class="stat-label">Avg Claim Amount</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">${formatMoney(d.avg_annual_premium)}</div>
                <div class="stat-label">Avg Annual Premium</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatMoney(d.total_exposure)}</div>
                <div class="stat-label">Total Exposure</div>
            </div>
        `;
    }

    // Claims by type chart
    const byType = await apiCall('/api/analytics/claims-by-type');
    if (byType?.data) {
        makeChart('chart-claims-type', {
            type: 'doughnut',
            data: {
                labels: Object.keys(byType.data),
                datasets: [{
                    data: Object.values(byType.data),
                    backgroundColor: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'],
                }]
            },
            options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
        });
    }

    // Claim amount distribution
    const amtDist = await apiCall('/api/analytics/claim-amount-distribution');
    if (amtDist?.data) {
        makeChart('chart-amount-dist', {
            type: 'bar',
            data: {
                labels: amtDist.data.labels,
                datasets: [
                    { label: 'Legitimate', data: amtDist.data.legitimate, backgroundColor: '#3b82f6' },
                    { label: 'Fraudulent', data: amtDist.data.fraudulent, backgroundColor: '#ef4444' },
                ]
            },
            options: {
                responsive: true,
                scales: { x: { stacked: false }, y: { beginAtZero: true } },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // Model metrics
    const metrics = await apiCall('/api/fraud/model-metrics');
    if (metrics?.data) {
        const m = metrics.data.metrics;
        const cm = metrics.data.confusion_matrix;
        $('model-metrics-container').innerHTML = `
            <div class="stats-grid">
                <div class="stat-card success">
                    <div class="stat-value">${formatPct(m.accuracy * 100)}</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card info">
                    <div class="stat-value">${formatPct(m.precision * 100)}</div>
                    <div class="stat-label">Precision</div>
                </div>
                <div class="stat-card warning">
                    <div class="stat-value">${formatPct(m.recall * 100)}</div>
                    <div class="stat-label">Recall</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${formatPct(m.f1_score * 100)}</div>
                    <div class="stat-label">F1 Score</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">${m.roc_auc.toFixed(3)}</div>
                    <div class="stat-label">ROC-AUC</div>
                </div>
            </div>
            <div style="margin-top:10px;">
                <strong>Confusion Matrix:</strong>
                <table class="data-table" style="max-width:400px;margin-top:8px;">
                    <tr><th></th><th>Predicted Legit</th><th>Predicted Fraud</th></tr>
                    <tr><td><strong>Actual Legit</strong></td><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr>
                    <tr><td><strong>Actual Fraud</strong></td><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr>
                </table>
                <p style="margin-top:8px;font-size:0.85rem;color:#6b7280;">
                    Training samples: ${metrics.data.training_samples} | Test samples: ${metrics.data.test_samples}
                </p>
            </div>
        `;
    }
}

// -------------------------------------------------------------------
// FRAUD DETECTION
// -------------------------------------------------------------------
$('btn-score-fraud').addEventListener('click', async () => {
    const btn = $('btn-score-fraud');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';

    const payload = {
        age: +$('f-age').value,
        credit_score: +$('f-credit').value,
        annual_premium: +$('f-premium').value,
        years_as_customer: +$('f-years').value,
        num_prior_claims: +$('f-prior').value,
        has_violations: +$('f-violations').value,
        claim_amount: +$('f-amount').value,
        num_witnesses: +$('f-witnesses').value,
        police_report_filed: +$('f-police').value,
        report_delay_days: +$('f-delay').value,
        claim_type: $('f-type').value,
        severity: $('f-severity').value,
        policy_type: $('f-policy-type').value,
        gender: $('f-gender').value,
    };

    const res = await apiCall('/api/fraud/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    btn.disabled = false;
    btn.innerHTML = '&#128269; Analyze Fraud Risk';

    if (res?.data) {
        const d = res.data;
        const riskClass = d.risk_level.toLowerCase();
        let explanationsHtml = d.explanations.map(e => `
            <div class="explanation-card">
                <div class="feat-name">${e.feature} (${e.importance_pct}% importance)</div>
                <div class="feat-detail">${e.explanation}</div>
            </div>
        `).join('');

        $('fraud-result-content').innerHTML = `
            <div class="stats-grid" style="margin-bottom:16px;">
                <div class="stat-card ${riskClass === 'high' ? 'danger' : riskClass === 'medium' ? 'warning' : 'success'}">
                    <div class="stat-value">${(d.fraud_probability * 100).toFixed(1)}%</div>
                    <div class="stat-label">Fraud Probability</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${d.risk_score}</div>
                    <div class="stat-label">Risk Score (0-100)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value"><span class="risk-badge ${riskClass}">${d.risk_level}</span></div>
                    <div class="stat-label">Risk Level</div>
                </div>
                <div class="stat-card ${d.anomaly_flag ? 'danger' : 'success'}">
                    <div class="stat-value">${d.anomaly_flag ? 'YES' : 'NO'}</div>
                    <div class="stat-label">Anomaly Detected</div>
                </div>
            </div>
            <h4 style="margin-bottom:10px;">Why This Prediction Was Made:</h4>
            ${explanationsHtml}
        `;
        $('fraud-result').style.display = 'block';
    }
});

async function loadFeatureImportances() {
    const res = await apiCall('/api/fraud/feature-importances');
    if (res?.data) {
        const labels = res.data.map(d => d.feature);
        const values = res.data.map(d => d.importance);
        makeChart('chart-feature-imp', {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Importance', data: values, backgroundColor: '#3b82f6' }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                scales: { x: { beginAtZero: true } },
                plugins: { legend: { display: false } }
            }
        });
    }
}

// -------------------------------------------------------------------
// WHAT-IF SCENARIO ANALYSIS
// -------------------------------------------------------------------
$('btn-what-if').addEventListener('click', async () => {
    const btn = $('btn-what-if');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running scenarios...';

    // Build payload from the current fraud form values
    const payload = {
        age: +$('f-age').value,
        credit_score: +$('f-credit').value,
        annual_premium: +$('f-premium').value,
        years_as_customer: +$('f-years').value,
        num_prior_claims: +$('f-prior').value,
        has_violations: +$('f-violations').value,
        claim_amount: +$('f-amount').value,
        num_witnesses: +$('f-witnesses').value,
        police_report_filed: +$('f-police').value,
        report_delay_days: +$('f-delay').value,
        claim_type: $('f-type').value,
        severity: $('f-severity').value,
        policy_type: $('f-policy-type').value,
        gender: $('f-gender').value,
    };

    const res = await apiCall('/api/fraud/what-if', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    btn.disabled = false;
    btn.innerHTML = '&#9881; Run What-If Scenarios';

    if (res?.data) {
        const d = res.data;
        let scenarioCards = d.scenarios.map(s => {
            const deltaSign = s.delta > 0 ? '+' : '';
            const deltaClass = s.delta > 0 ? 'positive' : s.delta < 0 ? 'negative' : 'neutral';
            return `
                <div class="whatif-card ${s.direction}">
                    <div class="scenario-label">${s.scenario}</div>
                    <div class="scenario-delta ${deltaClass}">${deltaSign}${s.delta} pts</div>
                    <div class="scenario-detail">New score: ${s.new_score} (${s.new_level})</div>
                </div>
            `;
        }).join('');

        $('whatif-content').innerHTML = `
            <div class="stats-grid" style="margin-bottom:16px;">
                <div class="stat-card info">
                    <div class="stat-value">${d.base_score}</div>
                    <div class="stat-label">Base Risk Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${d.base_level}</div>
                    <div class="stat-label">Base Risk Level</div>
                </div>
            </div>
            <h4 style="margin-bottom:12px;">How would changing factors affect the score?</h4>
            <div class="whatif-grid">${scenarioCards}</div>
        `;
        $('whatif-result').style.display = 'block';
    }
});

// -------------------------------------------------------------------
// RISK PROFILE COMPARISON
// -------------------------------------------------------------------
$('btn-risk-profile').addEventListener('click', async () => {
    const btn = $('btn-risk-profile');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Loading profiles...';

    const res = await apiCall('/api/fraud/risk-profile');

    btn.disabled = false;
    btn.innerHTML = '&#128202; Load Risk Profiles';

    if (res?.data) {
        const fp = res.data.fraud_profile;
        const lp = res.data.legitimate_profile;
        const diffs = res.data.key_differentiators;

        function profileStats(p) {
            return `
                <div class="profile-stat"><span class="label">Avg Age</span><span class="value">${p.avg_age}</span></div>
                <div class="profile-stat"><span class="label">Avg Credit Score</span><span class="value">${p.avg_credit_score}</span></div>
                <div class="profile-stat"><span class="label">Avg Claim Amount</span><span class="value">${formatMoney(p.avg_claim_amount)}</span></div>
                <div class="profile-stat"><span class="label">Avg Prior Claims</span><span class="value">${p.avg_prior_claims}</span></div>
                <div class="profile-stat"><span class="label">Avg Report Delay</span><span class="value">${p.avg_report_delay} days</span></div>
                <div class="profile-stat"><span class="label">Has Violations</span><span class="value">${p.pct_with_violations}%</span></div>
                <div class="profile-stat"><span class="label">No Police Report</span><span class="value">${p.pct_no_police_report}%</span></div>
                <div class="profile-stat"><span class="label">Total Cases</span><span class="value">${p.count}</span></div>
            `;
        }

        $('risk-profile-content').innerHTML = `
            <div class="profile-compare">
                <div class="profile-panel fraud">
                    <h4 style="color:var(--danger);">&#9888; Fraudulent Claims Profile</h4>
                    ${profileStats(fp)}
                </div>
                <div class="profile-panel legit">
                    <h4 style="color:var(--success);">&#9989; Legitimate Claims Profile</h4>
                    ${profileStats(lp)}
                </div>
            </div>
        `;
        $('risk-profile-result').style.display = 'block';

        // Chart comparing key differentiators
        const labels = diffs.map(d => d.factor);
        const fraudVals = diffs.map(d => d.fraud_avg);
        const legitVals = diffs.map(d => d.legit_avg);

        makeChart('chart-risk-comparison', {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    { label: 'Fraud Avg', data: fraudVals, backgroundColor: '#ef4444' },
                    { label: 'Legitimate Avg', data: legitVals, backgroundColor: '#10b981' },
                ]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } },
                plugins: {
                    legend: { position: 'bottom' },
                    title: { display: true, text: 'Key Differentiators: Fraud vs Legitimate' }
                }
            }
        });
    }
});

// -------------------------------------------------------------------
// CLAIMS PROCESSING
// -------------------------------------------------------------------
$('btn-process-claim').addEventListener('click', async () => {
    const btn = $('btn-process-claim');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Processing...';

    const payload = {
        description: $('c-description').value,
        claim_amount: +$('c-amount').value,
        fraud_probability: +$('c-fraud-prob').value,
        police_report: $('c-police').value === 'true',
    };

    const res = await apiCall('/api/claims/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    btn.disabled = false;
    btn.innerHTML = '&#128196; Process Claim';

    if (res?.data) {
        const d = res.data;
        const prClass = d.priority_routing.priority === 'Critical' ? 'high' :
                        d.priority_routing.priority === 'High' ? 'high' :
                        d.priority_routing.priority === 'Medium' ? 'medium' : 'low';

        // Build entities list
        let entitiesHtml = '';
        for (const [key, val] of Object.entries(d.entities)) {
            if (key === 'key_flags') continue;
            if (Array.isArray(val) && val.length > 0) {
                entitiesHtml += `<tr><td><strong>${key}</strong></td><td>${val.join(', ')}</td></tr>`;
            }
        }
        let flagsHtml = '';
        if (d.entities.key_flags) {
            for (const [k, v] of Object.entries(d.entities.key_flags)) {
                flagsHtml += `<tr><td>${k}</td><td>${v ? '&#9989; Yes' : '&#10060; No'}</td></tr>`;
            }
        }

        $('claims-result-content').innerHTML = `
            <div class="stats-grid" style="margin-bottom:16px;">
                <div class="stat-card info">
                    <div class="stat-value">${d.classification.predicted_type}</div>
                    <div class="stat-label">Predicted Type (${(d.classification.confidence*100).toFixed(0)}%)</div>
                </div>
                <div class="stat-card warning">
                    <div class="stat-value">${d.severity.predicted_severity}</div>
                    <div class="stat-label">Severity (${(d.severity.confidence*100).toFixed(0)}%)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value"><span class="risk-badge ${prClass}">${d.priority_routing.priority}</span></div>
                    <div class="stat-label">Priority (Score: ${d.priority_routing.priority_score})</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">${formatMoney(d.settlement_estimate.estimated_settlement)}</div>
                    <div class="stat-label">Est. Settlement</div>
                </div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div>
                    <h4 style="margin-bottom:8px;">Extracted Entities (NLP)</h4>
                    <table class="data-table">${entitiesHtml || '<tr><td>No entities found</td></tr>'}</table>
                </div>
                <div>
                    <h4 style="margin-bottom:8px;">Key Flags</h4>
                    <table class="data-table">${flagsHtml}</table>
                </div>
            </div>

            <div style="margin-top:16px;">
                <h4>Settlement Range</h4>
                <p style="font-size:0.95rem;margin-top:4px;">
                    ${formatMoney(d.settlement_estimate.range_low)} &mdash;
                    ${formatMoney(d.settlement_estimate.range_high)}
                </p>
                <p style="font-size:0.85rem;color:#6b7280;margin-top:4px;">
                    Routing: <strong>${d.priority_routing.routing}</strong>
                </p>
            </div>
        `;
        $('claims-result').style.display = 'block';
    }
});

// -------------------------------------------------------------------
// DATA QUALITY
// -------------------------------------------------------------------
$('btn-quality-report').addEventListener('click', async () => {
    const btn = $('btn-quality-report');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';

    const res = await apiCall('/api/data-quality/report');

    btn.disabled = false;
    btn.innerHTML = '&#9989; Generate Quality Report';

    if (res?.data) {
        const d = res.data;
        const s = d.summary;
        const barColor = s.health_score >= 80 ? '#10b981' : s.health_score >= 60 ? '#f59e0b' : '#ef4444';

        $('quality-summary-content').innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${s.total_records.toLocaleString()}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card ${s.health_score >= 80 ? 'success' : s.health_score >= 60 ? 'warning' : 'danger'}">
                    <div class="stat-value">${s.health_score}/100 (${s.health_grade})</div>
                    <div class="stat-label">Health Score</div>
                </div>
                <div class="stat-card danger">
                    <div class="stat-value">${s.critical_issues}</div>
                    <div class="stat-label">Critical Issues</div>
                </div>
                <div class="stat-card warning">
                    <div class="stat-value">${s.warnings}</div>
                    <div class="stat-label">Warnings</div>
                </div>
            </div>
            <div class="health-bar">
                <div class="health-bar-fill" style="width:${s.health_score}%;background:${barColor};">
                    ${s.health_score}%
                </div>
            </div>
        `;

        // Build issues list
        let issuesHtml = '';

        // Missing values
        for (const ds of [d.missing_values.claims, d.missing_values.policies]) {
            for (const issue of ds.issues) {
                issuesHtml += `
                    <div class="issue-item">
                        <span class="issue-badge ${issue.severity}">${issue.severity}</span>
                        <div>
                            <strong>${ds.dataset}.${issue.column}</strong>: ${issue.count} missing (${issue.percentage}%)
                            <div style="font-size:0.82rem;color:#6b7280;">${issue.recommendation}</div>
                        </div>
                    </div>`;
            }
        }

        // Duplicates
        for (const key of ['claims', 'policies']) {
            const dup = d.duplicates[key];
            if (dup.duplicate_rows > 0) {
                issuesHtml += `
                    <div class="issue-item">
                        <span class="issue-badge ${dup.severity}">${dup.severity}</span>
                        <div>
                            <strong>${key}</strong>: ${dup.duplicate_rows} duplicate rows (${dup.duplicate_groups} groups)
                            <div style="font-size:0.82rem;color:#6b7280;">${dup.recommendation}</div>
                        </div>
                    </div>`;
            }
        }

        // Consistency
        for (const issue of d.consistency) {
            issuesHtml += `
                <div class="issue-item">
                    <span class="issue-badge ${issue.severity}">${issue.severity}</span>
                    <div>
                        <strong>${issue.column}</strong>: ${issue.issue_type}
                        ${issue.bad_values ? ' (' + issue.bad_values.slice(0, 3).join(', ') + ')' : ''}
                        <div style="font-size:0.82rem;color:#6b7280;">${issue.recommendation}</div>
                    </div>
                </div>`;
        }

        $('quality-issues-content').innerHTML = issuesHtml || '<p>No issues found!</p>';
        $('quality-result').style.display = 'block';
        $('clean-result').style.display = 'none';
    }
});

$('btn-clean-data').addEventListener('click', async () => {
    const btn = $('btn-clean-data');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Cleaning...';

    const res = await apiCall('/api/data-quality/clean', { method: 'POST' });

    btn.disabled = false;
    btn.innerHTML = '&#128295; Clean & Standardize Data';

    if (res?.data) {
        const d = res.data;
        let fixesHtml = d.fixes_applied.map(f => `
            <div class="issue-item">
                <span class="issue-badge ok">fixed</span>
                <div>${f}</div>
            </div>
        `).join('');

        $('clean-result-content').innerHTML = `
            <div class="stats-grid" style="margin-bottom:16px;">
                <div class="stat-card">
                    <div class="stat-value">${d.original_records}</div>
                    <div class="stat-label">Original Records</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">${d.cleaned_records}</div>
                    <div class="stat-label">Cleaned Records</div>
                </div>
                <div class="stat-card info">
                    <div class="stat-value">${d.fixes_applied.length}</div>
                    <div class="stat-label">Fixes Applied</div>
                </div>
            </div>
            ${fixesHtml}
            <p style="margin-top:12px;font-size:0.85rem;color:#6b7280;">
                Output files: ${d.output_files.join(', ')}
            </p>
        `;
        $('clean-result').style.display = 'block';
    }
});

// -------------------------------------------------------------------
// ANALYTICS
// -------------------------------------------------------------------
let analyticsLoaded = false;

async function loadAnalytics() {
    if (analyticsLoaded) return;
    analyticsLoaded = true;

    // Monthly trends
    const trends = await apiCall('/api/analytics/monthly-trends');
    if (trends?.data) {
        // Sample every 3rd month to avoid overcrowding
        const sampled = trends.data.filter((_, i) => i % 3 === 0);
        makeChart('chart-monthly-trend', {
            type: 'line',
            data: {
                labels: sampled.map(d => d.month),
                datasets: [
                    {
                        label: 'Total Claims',
                        data: sampled.map(d => d.total_claims),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.1)',
                        fill: true, tension: 0.3,
                    },
                    {
                        label: 'Fraud Claims',
                        data: sampled.map(d => d.fraud_claims),
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239,68,68,0.1)',
                        fill: true, tension: 0.3,
                    }
                ]
            },
            options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
        });
    }

    // Fraud by type
    const fbt = await apiCall('/api/analytics/fraud-by-type');
    if (fbt?.data) {
        makeChart('chart-fraud-by-type', {
            type: 'bar',
            data: {
                labels: fbt.data.map(d => d.claim_type_clean),
                datasets: [
                    { label: 'Total', data: fbt.data.map(d => d.total), backgroundColor: '#3b82f6' },
                    { label: 'Fraud', data: fbt.data.map(d => d.fraud), backgroundColor: '#ef4444' },
                ]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // Severity
    const sev = await apiCall('/api/analytics/claims-by-severity');
    if (sev?.data) {
        makeChart('chart-severity-dist', {
            type: 'pie',
            data: {
                labels: Object.keys(sev.data),
                datasets: [{ data: Object.values(sev.data), backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6'] }]
            },
            options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
        });
    }

    // Risk distributions
    const risk = await apiCall('/api/analytics/risk-distribution');
    if (risk?.data) {
        // Age
        makeChart('chart-age-dist', {
            type: 'bar',
            data: {
                labels: Object.keys(risk.data.age_distribution),
                datasets: [{ label: 'Policyholders', data: Object.values(risk.data.age_distribution), backgroundColor: '#8b5cf6' }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
        });

        // Credit
        makeChart('chart-credit-dist', {
            type: 'bar',
            data: {
                labels: Object.keys(risk.data.credit_score_distribution),
                datasets: [{ label: 'Policyholders', data: Object.values(risk.data.credit_score_distribution), backgroundColor: '#10b981' }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
        });

        // States
        makeChart('chart-state-dist', {
            type: 'bar',
            data: {
                labels: Object.keys(risk.data.top_states),
                datasets: [{ label: 'Policies', data: Object.values(risk.data.top_states), backgroundColor: '#f59e0b' }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                scales: { x: { beginAtZero: true } },
                plugins: { legend: { display: false } }
            }
        });
    }
}

// -------------------------------------------------------------------
// Init - load dashboard on startup
// -------------------------------------------------------------------
loadDashboard();
