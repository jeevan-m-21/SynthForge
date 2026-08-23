/**
 * SynthForge — Main Application Controller (Phase 8 Streamlined UX)
 */
const state = {
  datasetId: null,
  jobId: null,
  datasetInfo: null,
  lastGenResult: null,
  lastQualityReport: null,
  loading: false,
  jobs: [],
  selectedMode: 'fast', // 'fast' | 'high'
};

// ── Navigation ──
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(t => {
    t.classList.remove('active');
    t.setAttribute('aria-hidden', 'true');
  });
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.remove('active');
    t.setAttribute('aria-selected', 'false');
  });

  const activeContent = document.getElementById(tabId);
  const activeTabBtn = document.querySelector(`[data-tab="${tabId}"]`);
  if (activeContent) {
    activeContent.classList.add('active');
    activeContent.setAttribute('aria-hidden', 'false');
  }
  if (activeTabBtn) {
    activeTabBtn.classList.add('active');
    activeTabBtn.setAttribute('aria-selected', 'true');
  }

  // Contextual loaders
  if (tabId === 'tab-privacy-threats') loadPrivacyBudget();
  if (tabId === 'tab-synthesize') loadJobHistory();
}

// ── Accordion Helper ──
function toggleAccordion(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('open');
}

// ── Toast Notifications ──
function toast(msg, type = 'success') {
  const c = document.getElementById('toast-container');
  if (!c) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✗' : '⚠'}</span><span>${msg}</span>`;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateX(100%)';
    t.style.transition = 'all 0.3s ease';
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

// ── Loading Overlay ──
function showLoading(msg) {
  state.loading = true;
  const overlay = document.getElementById('loading-overlay');
  const text = document.getElementById('loading-text');
  if (overlay) overlay.style.display = 'flex';
  if (text) text.textContent = msg || 'Processing...';
}
function hideLoading() {
  state.loading = false;
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.style.display = 'none';
}

// ── Generation Mode Selection ──
function selectGenerationMode(mode) {
  state.selectedMode = mode;
  const modeInput = document.getElementById('gen-selected-mode');
  if (modeInput) modeInput.value = mode;

  const fastCard = document.getElementById('mode-card-fast');
  const highCard = document.getElementById('mode-card-high');
  const modelSelect = document.getElementById('gen-model');

  if (mode === 'fast') {
    fastCard?.classList.add('selected');
    fastCard?.setAttribute('aria-checked', 'true');
    highCard?.classList.remove('selected');
    highCard?.setAttribute('aria-checked', 'false');
    if (modelSelect) modelSelect.value = 'statistical';
  } else {
    highCard?.classList.add('selected');
    highCard?.setAttribute('aria-checked', 'true');
    fastCard?.classList.remove('selected');
    fastCard?.setAttribute('aria-checked', 'false');
    if (modelSelect) modelSelect.value = 'tvae';
  }
}

// ── Dataset Upload & Sample Loading ──
async function handleFileUpload(file) {
  if (!file || !file.name.toLowerCase().endsWith('.csv')) {
    toast('Please select a valid CSV file.', 'error');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    toast('File size exceeds the 50 MB upload limit.', 'error');
    return;
  }

  showLoading('Uploading and profiling dataset schema...');
  try {
    const res = await api.uploadFile(file);
    // Reset previous generation state on new upload
    state.datasetId = res.data.dataset_id;
    state.datasetInfo = res.data;
    state.jobId = null;
    state.lastGenResult = null;
    state.lastQualityReport = null;

    renderDatasetInfo(res.data);
    toast(`Dataset uploaded: ${res.data.num_rows.toLocaleString()} rows, ${res.data.num_cols} columns`);
  } catch (e) {
    toast(e.message, 'error');
  }
  hideLoading();
}

async function loadSampleData() {
  showLoading('Loading sample healthcare dataset...');
  try {
    const res = await api.loadSample();
    // Reset previous generation state on sample load
    state.datasetId = res.data.dataset_id;
    state.datasetInfo = res.data;
    state.jobId = null;
    state.lastGenResult = null;
    state.lastQualityReport = null;

    renderDatasetInfo(res.data);
    toast(`Sample healthcare data loaded: ${res.data.num_rows.toLocaleString()} rows`);
  } catch (e) {
    toast(e.message, 'error');
  }
  hideLoading();
}

function renderDatasetInfo(data) {
  const el = document.getElementById('dataset-status');
  if (!el) return;

  el.innerHTML = `
    <div class="grid grid-3" style="margin-top:1.2rem">
      <div class="stat-card"><div class="stat-label">Selected Dataset</div><div class="stat-value blue" style="font-size:1.25rem">${data.filename}</div></div>
      <div class="stat-card"><div class="stat-label">Total Records</div><div class="stat-value green">${data.num_rows.toLocaleString()}</div></div>
      <div class="stat-card"><div class="stat-label">Attributes</div><div class="stat-value purple">${data.num_cols} Columns</div></div>
    </div>
    <div class="card" style="margin-top:1rem;background:rgba(255,255,255,0.02)">
      <div class="card-title" style="font-size:0.9rem">📋 Schema Attributes Preview</div>
      <div class="grid grid-4">${Object.entries(data.column_types || {}).map(([k, v]) =>
        `<div class="stat-card"><div class="stat-label">${k}</div><span class="badge badge-${v === 'numerical' ? 'info' : v === 'boolean' ? 'purple' : 'warning'}">${v}</span></div>`
      ).join('')}</div>
    </div>`;

  const genIdInput = document.getElementById('gen-dataset-id');
  if (genIdInput) genIdInput.value = data.dataset_id;

  const genRowsInput = document.getElementById('gen-rows');
  if (genRowsInput && data.num_rows) {
    genRowsInput.value = Math.min(data.num_rows, 100000);
  }
}

// ── Synthetic Data Generation (Goal-Oriented Flow) ──
async function generateData() {
  const dsId = document.getElementById('gen-dataset-id')?.value || state.datasetId;
  if (!dsId) {
    toast('Please upload or select a dataset first.', 'error');
    return;
  }

  const numRows = parseInt(document.getElementById('gen-rows')?.value, 10) || 1000;
  if (numRows < 10 || numRows > 100000) {
    toast('Number of rows must be between 10 and 100,000.', 'error');
    return;
  }

  // Determine model based on mode vs explicit override
  const mode = state.selectedMode;
  let modelType = mode === 'fast' ? 'statistical' : 'tvae';

  // Check if advanced override was explicitly set
  const modelOverride = document.getElementById('gen-model')?.value;
  if (modelOverride && modelOverride !== 'statistical' && mode === 'fast') {
    modelType = modelOverride;
  } else if (modelOverride && mode === 'high') {
    modelType = modelOverride;
  }

  const epochs = parseInt(document.getElementById('gen-epochs')?.value, 10) || 50;
  const batchSize = parseInt(document.getElementById('gen-batch-size')?.value, 10) || 500;
  const epsilon = parseFloat(document.getElementById('gen-epsilon')?.value) || 1.0;
  const delta = parseFloat(document.getElementById('gen-delta')?.value) || 1e-5;
  const dpMechanism = document.getElementById('gen-mechanism')?.value || 'gaussian';
  const applyDp = document.getElementById('gen-apply-dp')?.checked ?? true;

  const rawSeed = document.getElementById('gen-seed')?.value?.trim();
  const seed = rawSeed !== '' && rawSeed !== undefined ? parseInt(rawSeed, 10) : undefined;

  const loadingMsg = mode === 'fast'
    ? 'Generating synthetic data with Fast Statistical Copula modeling...'
    : 'Training deep generative model (TVAE)... This may take 1-2 minutes on CPU.';
  showLoading(loadingMsg);

  const btn = document.getElementById('btn-generate');
  if (btn) btn.disabled = true;

  try {
    const payload = {
      dataset_id: dsId,
      num_rows: numRows,
      model_type: modelType,
      epochs: epochs,
      batch_size: batchSize,
      epsilon: epsilon,
      delta: delta,
      dp_mechanism: dpMechanism,
      apply_dp: applyDp,
    };
    if (seed !== undefined && !isNaN(seed)) {
      payload.seed = seed;
    }

    const res = await api.generate(payload);
    state.jobId = res.data.job_id;
    state.lastGenResult = res.data;

    toast(`Successfully generated ${res.data.num_rows_generated.toLocaleString()} synthetic records!`);

    // Phase 8 Workflow: Automatically switch to Quality Report and evaluate
    switchTab('tab-quality');
    await runQualityEvaluation();
  } catch (e) {
    toast(e.message, 'error');
    hideLoading();
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ── Unified Quality Report (Primary Result Screen) ──
async function runQualityEvaluation() {
  if (!state.datasetId) {
    toast('Upload a dataset and generate synthetic data first.', 'error');
    return;
  }

  showLoading('Evaluating synthetic data quality and privacy trustworthiness...');
  try {
    const res = await api.getQualityReport({
      dataset_id: state.datasetId,
      synthetic_job_id: state.jobId || undefined,
    });
    state.lastQualityReport = res.data;
    renderQualityReport(res.data);
    toast('Quality Report evaluated successfully!');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    hideLoading();
  }
}

function renderQualityReport(report) {
  const el = document.getElementById('quality-results');
  if (!el || !report) return;

  const s = report.executive_summary || {};
  const structural = report.structural_fidelity || {};
  const statistical = report.statistical_fidelity || {};
  const relationship = report.relationship_fidelity || {};
  const ml = report.ml_utility || {};
  const privacy = report.privacy_risk || {};
  const warnings = report.all_warnings || [];

  const gen = state.lastGenResult || {};
  const rows = gen.num_rows_generated ? gen.num_rows_generated.toLocaleString() : 'Ready';
  const model = (gen.model_type || 'Synthetic').toUpperCase();
  const downloadUrl = state.jobId ? api.downloadUrl(state.jobId) : '#';

  // Trust Verdict
  let verdictClass = 'ready';
  let verdictIcon = '✓';
  if (s.trust_verdict?.includes('Review Required')) {
    verdictClass = 'review';
    verdictIcon = '⚠';
  } else if (s.trust_verdict?.includes('Revision Required')) {
    verdictClass = 'revision';
    verdictIcon = '✕';
  }

  el.innerHTML = `
    <!-- Top Executive Banner with Primary Download Action -->
    <div class="card" style="border:1px solid rgba(59,130,246,0.3);background:linear-gradient(135deg, rgba(17,24,39,0.85), rgba(15,23,42,0.95))">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1.5rem">
        <div style="flex:1;min-width:280px">
          <div class="verdict-banner ${verdictClass}" style="margin-bottom:0.5rem">
            <span style="font-size:1.4rem">${verdictIcon}</span>
            <span style="font-size:1.05rem;font-weight:700">${s.trust_verdict || 'Evaluation Complete'}</span>
          </div>
          <p style="color:var(--text-secondary);font-size:0.85rem;margin:0">
            ${gen.reproducible_run ? `<span class="badge badge-reproducible">✓ Reproducible (Seed: ${gen.seed})</span> • ` : ''}
            Model: <strong>${model}</strong> • Volume: <strong>${rows} rows</strong>
          </p>
        </div>

        <div>
          <a href="${downloadUrl}" class="btn btn-success btn-lg" download style="display:inline-flex;align-items:center;gap:8px;font-size:1rem;padding:12px 24px" ${!state.jobId ? 'disabled' : ''}>
            <span>⬇</span> <span>Download Synthetic Dataset</span>
          </a>
        </div>
      </div>
    </div>

    <!-- Dual Dimension Scorecard: Fidelity vs. Privacy -->
    <div class="grid grid-2" style="margin-top:1.5rem">
      <!-- Dimension 1: Data Fidelity -->
      <div class="card" style="border-top:4px solid var(--accent-blue)">
        <div class="card-title">📊 Dimension 1: Data Fidelity</div>
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">Evaluates how faithfully the synthetic records preserve empirical clinical distributions, statistical moments, and multi-variable correlations.</p>
        <div class="grid grid-2">
          <div class="stat-card"><div class="stat-label">Fidelity Score</div><div class="stat-value blue">${s.data_fidelity_score ?? '-'}/100</div></div>
          <div class="stat-card"><div class="stat-label">Fidelity Grade</div><div class="grade grade-${s.data_fidelity_grade || 'B'}">${s.data_fidelity_grade || '-'}</div></div>
        </div>
      </div>

      <!-- Dimension 2: Privacy Protection -->
      <div class="card" style="border-top:4px solid var(--accent-green)">
        <div class="card-title">🔐 Dimension 2: Privacy Protection</div>
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">Evaluates empirical protection against Membership Inference, Re-Identification, Attribute Leakage, and exact training record collisions.</p>
        <div class="grid grid-2">
          <div class="stat-card"><div class="stat-label">Protection Score</div><div class="stat-value green">${s.privacy_protection_score ?? '-'}/100</div></div>
          <div class="stat-card"><div class="stat-label">Privacy Grade</div><div class="grade grade-${s.privacy_protection_grade || 'A'}">${s.privacy_protection_grade || '-'}</div></div>
        </div>
      </div>
    </div>

    <!-- 5-Pillar Score Summary -->
    <div class="card" style="margin-top:1.5rem">
      <div class="card-title">🏛️ 5-Pillar Trustworthiness Breakdown</div>
      <div class="grid grid-3" style="margin-top:1rem">
        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">1. Structural Fidelity</div>
            <span class="badge badge-${structural.status === 'passed' ? 'success' : 'warning'}">${structural.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-blue)">${structural.score?.toFixed(1) ?? '-'}</div>
          <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">Schema completeness & datatype match</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">2. Statistical Fidelity</div>
            <span class="badge badge-${statistical.status === 'passed' ? 'success' : 'warning'}">${statistical.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-cyan)">${statistical.score?.toFixed(1) ?? '-'}</div>
          <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">KS distribution conformity & correlations</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">3. Relationship Fidelity</div>
            <span class="badge badge-${relationship.status === 'passed' ? 'success' : 'warning'}">${relationship.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-purple)">${relationship.score?.toFixed(1) ?? '-'}</div>
          <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">Multi-variable covariance alignment</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">4. ML Utility (TSTR)</div>
            <span class="badge badge-${ml.status === 'passed' ? 'success' : 'warning'}">${ml.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-amber)">${ml.score?.toFixed(1) ?? '-'}</div>
          <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">Train on Synth, Test on Real parity</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">5. Privacy Protection</div>
            <span class="badge badge-${privacy.status === 'passed' ? 'success' : 'warning'}">${privacy.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-green)">${privacy.privacy_protection_score?.toFixed(1) ?? '-'}</div>
          <p style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.4rem">MIA defense & zero collisions</p>
        </div>
      </div>
    </div>

    <!-- TECHNICAL DEEP-DIVE ACCORDIONS -->

    <!-- Accordion 1: Statistical Fidelity Details -->
    <div class="accordion" id="acc-stat-details" style="margin-top:1.5rem">
      <div class="accordion-header" onclick="toggleAccordion('acc-stat-details')">
        <span style="font-size:0.95rem;font-weight:700">📊 Statistical Fidelity & Distribution Analysis</span>
        <span class="accordion-icon">▼</span>
      </div>
      <div class="accordion-body">
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">
          <strong>What this measures:</strong> How closely synthetic values follow the empirical probability distributions and correlations found in the original clinical data.
        </p>

        ${statistical.column_reports ? `
        <div class="table-responsive">
          <table class="data-table">
            <thead><tr><th>Attribute</th><th>Type</th><th>KS / Chi2 Statistic</th><th>P-Value</th><th>Distribution Match</th><th>Quality</th></tr></thead>
            <tbody>${Object.entries(statistical.column_reports || {}).map(([col, r]) => {
              const test = r.ks_test || r.chi_squared || {};
              return `<tr><td>${col}</td><td><span class="badge badge-${r.type === 'numerical' ? 'info' : 'warning'}">${r.type}</span></td>
              <td>${test.statistic?.toFixed(4) ?? '-'}</td><td>${test.p_value?.toFixed(4) ?? '-'}</td>
              <td>${test.similar ? '<span class="badge badge-success">Match</span>' : '<span class="badge badge-danger">Divergent</span>'}</td>
              <td>${r.quality_score?.toFixed(1) ?? '-'}</td></tr>`;
            }).join('')}</tbody>
          </table>
        </div>` : '<p style="color:var(--text-muted)">No per-column statistical metrics available.</p>'}

        ${statistical.correlation?.columns?.length > 1 ? `
        <div class="grid grid-2" style="margin-top:1rem">
          <div class="card"><div class="card-title">🔵 Real Correlation Matrix</div><div id="qr-heatmap-real" class="heatmap-container"></div></div>
          <div class="card"><div class="card-title">🟣 Synthetic Correlation Matrix</div><div id="qr-heatmap-synth" class="heatmap-container"></div></div>
        </div>` : ''}

        <div class="grid grid-2" id="qr-dist-charts-container" style="margin-top:1rem"></div>
      </div>
    </div>

    <!-- Accordion 2: ML Utility (TSTR) Details -->
    <div class="accordion" id="acc-ml-details">
      <div class="accordion-header" onclick="toggleAccordion('acc-ml-details')">
        <span style="font-size:0.95rem;font-weight:700">🤖 Machine Learning Utility (TSTR Benchmark)</span>
        <span class="accordion-icon">▼</span>
      </div>
      <div class="accordion-body">
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">
          <strong>What this measures:</strong> Can predictive ML models trained exclusively on synthetic data generalize effectively when tested on holdout real patients?
          (<strong>TSTR</strong> = Train on Synthetic, Test on Real vs. <strong>TRTR</strong> = Train on Real, Test on Real).
        </p>

        ${ml.results ? `
        <div class="table-responsive">
          <table class="data-table">
            <thead><tr><th>Metric</th><th>Real Baseline (RF)</th><th>Synthetic Trained (RF)</th><th>Real Baseline (GB)</th><th>Synthetic Trained (GB)</th></tr></thead>
            <tbody>
              ${['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall'].map(m => `<tr><td><strong>${m.toUpperCase()}</strong></td>
                <td>${ml.results.trtr_rf?.[m] ?? '-'}</td><td>${ml.results.tstr_rf?.[m] ?? '-'}</td>
                <td>${ml.results.trtr_gb?.[m] ?? '-'}</td><td>${ml.results.tstr_gb?.[m] ?? '-'}</td></tr>`).join('')}
            </tbody>
          </table>
        </div>
        <div class="grid grid-2" style="margin-top:1rem">
          <div class="card"><div class="card-title">📊 ML Performance Parity</div><div class="chart-container"><canvas id="qr-ml-bar"></canvas></div></div>
          <div class="card"><div class="card-title">📈 ROC Curve (TSTR)</div><div class="chart-container"><canvas id="qr-ml-roc"></canvas></div></div>
        </div>` : '<p style="color:var(--text-muted)">No ML utility validation benchmark results available for this run.</p>'}
      </div>
    </div>

    <!-- Accordion 3: Structural & Privacy Details -->
    <div class="accordion" id="acc-struct-details">
      <div class="accordion-header" onclick="toggleAccordion('acc-struct-details')">
        <span style="font-size:0.95rem;font-weight:700">🏛️ Structural Integrity & Collision Defense</span>
        <span class="accordion-icon">▼</span>
      </div>
      <div class="accordion-body">
        <div class="grid grid-3">
          <div class="stat-card"><div class="stat-label">Columns Preserved</div><div class="stat-value green">${(structural.metrics?.column_preservation_rate * 100)?.toFixed(0) ?? 100}%</div></div>
          <div class="stat-card"><div class="stat-label">Exact Record Collisions</div><div class="stat-value ${privacy.metrics?.exact_duplicate_count === 0 ? 'green' : 'red'}">${privacy.metrics?.exact_duplicate_count ?? 0}</div></div>
          <div class="stat-card"><div class="stat-label">Covariance Alignment</div><div class="stat-value purple">${(relationship.metrics?.covariance_similarity * 100)?.toFixed(0) ?? '-'}%</div></div>
        </div>
      </div>
    </div>

    <!-- Findings & Warnings -->
    ${warnings.length ? `
    <div class="card" style="margin-top:1.5rem">
      <div class="card-title">⚠️ Quality & Privacy Findings (${warnings.length})</div>
      <ul style="padding-left:1.5rem;color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem">
        ${warnings.map(w => `<li style="margin-bottom:0.4rem">${w}</li>`).join('')}
      </ul>
    </div>` : ''}

    <div class="card" style="margin-top:1rem;background:rgba(255,255,255,0.02)">
      <p style="font-size:0.75rem;color:var(--text-muted);font-style:italic">
        ${s.disclaimer || 'Synthetic data trustworthiness cannot be determined by any single metric. Fidelity and Privacy represent separate trade-off dimensions.'}
      </p>
    </div>`;

  // Render heatmaps & distribution charts
  if (statistical.correlation?.columns?.length > 1) {
    setTimeout(() => {
      charts.heatmapHTML('qr-heatmap-real', statistical.correlation.real_correlation, statistical.correlation.columns);
      charts.heatmapHTML('qr-heatmap-synth', statistical.correlation.synth_correlation, statistical.correlation.columns);
    }, 100);
  }

  // Render distributions
  if (statistical.column_reports) {
    setTimeout(() => {
      const container = document.getElementById('qr-dist-charts-container');
      if (!container) return;
      let html = '';
      let idx = 0;
      for (const [col, report] of Object.entries(statistical.column_reports)) {
        if (report.distribution && (report.distribution.bins || report.distribution.categories)) {
          html += `<div class="card"><div class="chart-container"><canvas id="qr-dist-${idx}"></canvas></div></div>`;
        }
        idx++;
      }
      container.innerHTML = html;

      let i = 0;
      for (const [col, report] of Object.entries(statistical.column_reports)) {
        if (report.distribution) {
          if (report.type === 'numerical' && report.distribution.bins) charts.distribution(`qr-dist-${i}`, report.distribution, col);
          else if (report.distribution.categories) charts.categoricalDist(`qr-dist-${i}`, report.distribution, col);
        }
        i++;
      }
    }, 150);
  }

  // Render ML charts
  if (ml.results) {
    setTimeout(() => {
      const r = ml.results;
      if (r.trtr_rf && r.tstr_rf) {
        charts.comparisonBar('qr-ml-bar', ['Accuracy', 'F1', 'AUC'],
          [r.trtr_rf.accuracy, r.trtr_rf.f1_score, r.trtr_rf.roc_auc],
          [r.tstr_rf.accuracy, r.tstr_rf.f1_score, r.tstr_rf.roc_auc]);
      }
      if (r.tstr_rf?.roc_curve) {
        charts.rocCurve('qr-ml-roc', r.tstr_rf.roc_curve.fpr, r.tstr_rf.roc_curve.tpr, 'TSTR ROC');
      }
    }, 200);
  }
}

// ── Privacy & Threat Analysis Tab ──
async function loadPrivacyBudget() {
  const el = document.getElementById('privacy-content');
  if (!el) return;
  if (!state.datasetId) {
    el.innerHTML = '<div class="empty-state"><div class="empty-icon">🔐</div><p>Upload a dataset and generate synthetic records to view cumulative privacy budget expenditure.</p></div>';
    return;
  }
  try {
    const res = await api.getPrivacyBudget(state.datasetId);
    const b = res.data;
    el.innerHTML = `
      <div class="grid grid-2">
        <div class="card" style="text-align:center">
          <div class="card-title">🔐 Differential Privacy Budget</div>
          <div class="gauge-container">
            <canvas id="privacy-gauge"></canvas>
            <div class="gauge-label">
              <div class="gauge-value" style="color:${b.utilization_pct > 75 ? '#ef4444' : '#10b981'}">${b.utilization_pct}%</div>
              <div class="gauge-subtitle">Budget Utilized</div>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">📊 Budget Allocation Details</div>
          <div class="grid grid-2">
            <div class="stat-card"><div class="stat-label">ε Consumed</div><div class="stat-value amber">${b.total_epsilon_used}</div></div>
            <div class="stat-card"><div class="stat-label">ε Remaining</div><div class="stat-value green">${b.remaining_epsilon}</div></div>
            <div class="stat-card"><div class="stat-label">Max Budget Allowance</div><div class="stat-value blue">${b.max_epsilon}</div></div>
            <div class="stat-card"><div class="stat-label">Total Queries</div><div class="stat-value purple">${b.num_queries}</div></div>
          </div>
          ${b.warning_level ? `<div class="badge badge-danger" style="margin-top:1rem;display:inline-block">⚠ Notice: ${b.warning_level.toUpperCase()} budget utilization threshold reached</div>` : ''}
        </div>
      </div>
      ${b.history && b.history.length ? `
      <div class="card" style="margin-top:1.5rem">
        <div class="card-title">📜 Query History Log</div>
        <div class="table-responsive">
          <table class="data-table">
            <thead><tr><th>Operation</th><th>ε Spent</th><th>Timestamp</th></tr></thead>
            <tbody>${b.history.map(h => `<tr><td>${h.operation}</td><td>${h.epsilon}</td><td>${h.timestamp ? new Date(h.timestamp).toLocaleString() : '-'}</td></tr>`).join('')}</tbody>
          </table>
        </div>
      </div>` : ''}`;

    setTimeout(() => {
      const color = b.utilization_pct > 75 ? 'rgba(239,68,68,1)' : b.utilization_pct > 50 ? 'rgba(245,158,11,1)' : 'rgba(16,185,129,1)';
      charts.gauge('privacy-gauge', b.utilization_pct, 100, color);
    }, 100);
  } catch (e) {
    console.error(e);
  }
}

async function runAttacks() {
  if (!state.datasetId) {
    toast('Please upload a dataset and generate synthetic data first.', 'error');
    return;
  }
  showLoading('Simulating adversarial privacy attacks (MIA, Re-ID, Attribute Inference)...');
  try {
    const res = await api.simulateAttacks({ dataset_id: state.datasetId, synthetic_job_id: state.jobId || undefined });
    renderAttackResults(res.data);
    toast(`Overall Privacy Risk: ${res.data.overall_risk_score}/100 (${res.data.overall_risk_level})`);
  } catch (e) {
    toast(e.message, 'error');
  }
  hideLoading();
}

function renderAttackResults(data) {
  const el = document.getElementById('attack-results');
  if (!el) return;
  const a = data.attacks || {};
  const rl = l => `risk-${l}`;

  el.innerHTML = `
    <div class="grid grid-4" style="margin-top:1rem">
      <div class="stat-card"><div class="stat-label">Overall Risk Score</div><div class="stat-value ${rl(data.overall_risk_level)}">${data.overall_risk_score}</div></div>
      <div class="stat-card"><div class="stat-label">MIA Vulnerability</div><div class="stat-value ${rl(a.membership_inference?.risk_level)}">${a.membership_inference?.risk_score ?? '-'}</div></div>
      <div class="stat-card"><div class="stat-label">Re-ID Risk</div><div class="stat-value ${rl(a.reidentification?.risk_level)}">${a.reidentification?.risk_score ?? '-'}</div></div>
      <div class="stat-card"><div class="stat-label">Attribute Inference</div><div class="stat-value ${rl(a.attribute_inference?.risk_level)}">${a.attribute_inference?.risk_score ?? '-'}</div></div>
    </div>
    <div class="grid grid-2" style="margin-top:1.5rem">
      <div class="card"><div class="card-title">🎯 Threat Radar</div><div class="chart-container"><canvas id="attack-radar"></canvas></div></div>
      <div class="card"><div class="card-title">📈 MIA ROC Curve</div><div class="chart-container"><canvas id="attack-roc"></canvas></div></div>
    </div>
    <div class="card">
      <div class="card-title">🔍 Attack Vulnerability Summary</div>
      <div class="table-responsive">
        <table class="data-table">
          <thead><tr><th>Attack Vector</th><th>Metric</th><th>Empirical Value</th><th>Risk Assessment</th></tr></thead>
          <tbody>
            <tr><td>Membership Inference (MIA)</td><td>AUC</td><td>${a.membership_inference?.attack_auc ?? '-'}</td><td><span class="badge badge-${a.membership_inference?.risk_level === 'low' ? 'success' : 'danger'}">${a.membership_inference?.risk_level ?? '-'}</span></td></tr>
            <tr><td>Re-Identification</td><td>Records at Risk</td><td>${a.reidentification?.records_at_risk_pct ?? '-'}%</td><td><span class="badge badge-${a.reidentification?.risk_level === 'low' ? 'success' : 'danger'}">${a.reidentification?.risk_level ?? '-'}</span></td></tr>
            <tr><td>Attribute Inference</td><td>Average Advantage</td><td>${a.attribute_inference?.average_advantage ?? '-'}</td><td><span class="badge badge-${a.attribute_inference?.risk_level === 'low' ? 'success' : 'danger'}">${a.attribute_inference?.risk_level ?? '-'}</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div class="card"><p style="color:var(--text-secondary);font-size:0.9rem">${data.summary || ''}</p></div>`;

  setTimeout(() => {
    if (data.radar_chart) {
      charts.radar('attack-radar', data.radar_chart.labels, data.radar_chart.risk_scores);
    }
    if (a.membership_inference?.roc_curve) {
      charts.rocCurve('attack-roc', a.membership_inference.roc_curve.fpr, a.membership_inference.roc_curve.tpr, 'MIA');
    }
  }, 200);
}

// ── Job History & Recovery ──
async function loadJobHistory() {
  const container = document.getElementById('recent-jobs-section');
  if (!container) return;

  try {
    const res = await api.listJobs();
    const jobs = (res.data || []).sort((a, b) => (b.created_at || '').localeCompare(a.created_at || '')).slice(0, 5);
    state.jobs = jobs;

    if (!jobs.length) {
      container.innerHTML = '';
      return;
    }

    container.innerHTML = `
      <div class="card">
        <div class="card-title" style="font-size:0.95rem">📜 Recent Generation Jobs</div>
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Model</th>
                <th>Rows</th>
                <th>DP ε</th>
                <th>Reproducibility</th>
                <th>Time</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              ${jobs.map(j => {
                const statusBadge = getStatusBadge(j.status);
                const params = j.params || {};
                const model = (params.model_type || j.result?.model_type || 'statistical').toUpperCase();
                const rows = (params.num_rows || j.result?.num_rows_generated || 0).toLocaleString();
                const eps = params.epsilon ?? (j.result?.dp_metadata?.epsilon_actual ?? '-');
                const isRepro = params.seed !== undefined || j.result?.reproducible_run;
                const reproHtml = isRepro ? `<span class="badge badge-reproducible">Seed: ${params.seed ?? j.result?.seed}</span>` : '<span style="color:var(--text-muted)">-</span>';
                const created = j.created_at ? new Date(j.created_at).toLocaleTimeString() : '-';

                let actionHtml = '-';
                if (j.status === 'completed' && j.id) {
                  actionHtml = `<a href="${api.downloadUrl(j.id)}" class="btn btn-sm btn-secondary" download>Download</a>`;
                } else if (j.status === 'interrupted' || j.status === 'failed') {
                  actionHtml = `<button class="btn btn-sm btn-secondary" onclick="retryJob('${j.id}')">🔄 Retry</button>`;
                }

                return `
                  <tr>
                    <td>${statusBadge}</td>
                    <td>${model}</td>
                    <td>${rows}</td>
                    <td>${eps}</td>
                    <td>${reproHtml}</td>
                    <td>${created}</td>
                    <td>${actionHtml}</td>
                  </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  } catch (e) {
    console.warn('Could not load job history:', e);
  }
}

function getStatusBadge(status) {
  switch (status) {
    case 'completed': return '<span class="badge badge-success">✓ Completed</span>';
    case 'running': return '<span class="badge badge-running">● Running</span>';
    case 'pending': return '<span class="badge badge-pending">⏳ Pending</span>';
    case 'interrupted': return '<span class="badge badge-interrupted">⚠ Interrupted</span>';
    case 'failed': return '<span class="badge badge-danger">✗ Failed</span>';
    default: return `<span class="badge badge-info">${status || 'Unknown'}</span>`;
  }
}

function retryJob(jobId) {
  const job = state.jobs.find(j => j.id === jobId);
  if (!job || !job.params) {
    toast('Cannot find saved parameters for resubmission.', 'error');
    return;
  }
  const p = job.params;
  if (p.num_rows && document.getElementById('gen-rows')) document.getElementById('gen-rows').value = p.num_rows;
  if (p.model_type && document.getElementById('gen-model')) document.getElementById('gen-model').value = p.model_type;
  if (p.epochs && document.getElementById('gen-epochs')) document.getElementById('gen-epochs').value = p.epochs;
  if (p.epsilon && document.getElementById('gen-epsilon')) document.getElementById('gen-epsilon').value = p.epsilon;
  if (p.seed !== undefined && document.getElementById('gen-seed')) document.getElementById('gen-seed').value = p.seed;

  switchTab('tab-synthesize');
  toast('Pre-filled generation parameters. Ready for submission.');
}

// ── Application Initialization ──
document.addEventListener('DOMContentLoaded', () => {
  // Tab navigation bindings
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      switchTab(tab.dataset.tab);
    });
  });

  // File upload drag & drop
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  if (uploadZone) {
    uploadZone.addEventListener('dragover', e => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      if (e.dataTransfer.files?.[0]) handleFileUpload(e.dataTransfer.files[0]);
    });
  }
  if (fileInput) {
    fileInput.addEventListener('change', e => {
      if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
    });
  }
});
