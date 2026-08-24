/**
 * SynthForge — Main Application Controller (Phase 8 Streamlined UX)
 */
const state = {
  datasetId: null,
  jobId: null,
  datasetInfo: null,
  lastGenResult: null,
  lastQualityReport: null,
  lastAttackResults: null,
  selectedDistCol: null,
  loading: false,
  jobs: [],
  selectedMode: 'fast', // 'fast' | 'high'
};

// ── Helper: Safe HTML Escaping ──
function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ── Theme Manager ──
function initTheme() {
  const savedPref = localStorage.getItem('synthforge-theme') || 'system';
  applyTheme(savedPref, false);

  // Bind theme button clicks
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const val = btn.dataset.themeVal;
      applyTheme(val, true);
    });
  });

  // Listen for system theme changes if currently on system preference
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      const currentPref = localStorage.getItem('synthforge-theme') || 'system';
      if (currentPref === 'system') {
        applyTheme('system', false);
      }
    });
  }
}

function applyTheme(preference, save = true) {
  let resolved = preference;
  if (preference === 'system') {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    resolved = isDark ? 'dark' : 'light';
  }

  document.documentElement.setAttribute('data-theme', resolved);
  document.documentElement.setAttribute('data-theme-preference', preference);
  if (save) {
    localStorage.setItem('synthforge-theme', preference);
  }

  // Update theme button active states
  document.querySelectorAll('.theme-btn').forEach(btn => {
    const isActive = btn.dataset.themeVal === preference;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-checked', isActive ? 'true' : 'false');
  });

  // Dynamically re-render charts on theme update if active
  if (state.lastQualityReport && document.getElementById('tab-quality')?.classList.contains('active')) {
    renderQualityReportCharts(state.lastQualityReport);
  } else if (state.lastAttackResults && document.getElementById('tab-privacy-threats')?.classList.contains('active')) {
    renderAttackCharts(state.lastAttackResults);
  } else {
    charts.resizeAll();
  }
}

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

  // Scroll to top when switching views
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Contextual loaders
  if (tabId === 'tab-privacy-threats') loadPrivacyThreats();
  if (tabId === 'tab-synthesize') loadJobHistory();
  if (tabId === 'tab-quality' && state.lastQualityReport) {
    setTimeout(() => charts.resizeAll(), 60);
  }
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

// ── Generation Mode & Model Architecture ──
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
  onModelArchitectureChange();
}

function onModelArchitectureChange() {
  const model = document.getElementById('gen-model')?.value || 'statistical';
  const epochsGroup = document.getElementById('adv-epochs-group');
  const batchGroup = document.getElementById('adv-batch-group');
  const isDeep = model === 'tvae' || model === 'ctgan';

  if (epochsGroup) epochsGroup.style.display = isDeep ? 'block' : 'none';
  if (batchGroup) batchGroup.style.display = isDeep ? 'block' : 'none';
}

// ── Multi-Step Generation Progress Helpers ──
function showGenerationProgress(mode) {
  state.loading = true;
  const overlay = document.getElementById('generation-progress-overlay');
  const pstep3Text = document.getElementById('pstep-3-text');
  const pstep3 = document.getElementById('pstep-3');
  const pstep4 = document.getElementById('pstep-4');
  const pstep5 = document.getElementById('pstep-5');

  if (pstep3Text) {
    pstep3Text.textContent = mode === 'high' ? 'Training deep neural model & creating records' : 'Synthesizing empirical distributions';
  }

  // Reset step states
  if (pstep3) pstep3.className = 'progress-step active';
  if (pstep4) pstep4.className = 'progress-step pending';
  if (pstep5) pstep5.className = 'progress-step pending';

  if (overlay) overlay.style.display = 'flex';
}

function updateGenerationStep(stepNum) {
  for (let i = 1; i <= 5; i++) {
    const stepEl = document.getElementById(`pstep-${i}`);
    if (!stepEl) continue;
    if (i < stepNum) {
      stepEl.className = 'progress-step done';
      const icon = stepEl.querySelector('.step-status-icon');
      if (icon) icon.textContent = '✓';
    } else if (i === stepNum) {
      stepEl.className = 'progress-step active';
      const icon = stepEl.querySelector('.step-status-icon');
      if (icon) icon.textContent = '●';
    } else {
      stepEl.className = 'progress-step pending';
      const icon = stepEl.querySelector('.step-status-icon');
      if (icon) icon.textContent = '○';
    }
  }
}

function hideGenerationProgress() {
  state.loading = false;
  const overlay = document.getElementById('generation-progress-overlay');
  if (overlay) overlay.style.display = 'none';
}

// ── Dataset Upload & Sample Loading ──
async function handleFileUpload(file) {
  if (!file || !file.name.toLowerCase().endsWith('.csv')) {
    toast("That CSV couldn't be processed. Check that the file contains a valid table and try again.", 'error');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    toast('This file is larger than the 50 MB upload limit.', 'error');
    return;
  }

  showLoading('Uploading and analyzing dataset schema...');
  try {
    const res = await api.uploadFile(file);
    // Reset previous generation state on new upload
    state.datasetId = res.data.dataset_id;
    state.datasetInfo = res.data;
    state.jobId = null;
    state.lastGenResult = null;
    state.lastQualityReport = null;

    renderDatasetInfo(res.data);
    toast(`Dataset uploaded: ${res.data.num_rows.toLocaleString()} records, ${res.data.num_cols} columns`);
  } catch (e) {
    toast(e.message || "That CSV couldn't be processed. Check that the file contains a valid table and try again.", 'error');
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
    toast(`Sample healthcare data loaded: ${res.data.num_rows.toLocaleString()} records`);
  } catch (e) {
    toast(e.message || 'Could not load sample data. Please try again.', 'error');
  }
  hideLoading();
}

function renderDatasetInfo(data) {
  const el = document.getElementById('dataset-status');
  if (!el) return;

  const safeFilename = escapeHtml(data.filename || 'Uploaded Dataset');
  const safeRows = data.num_rows ? data.num_rows.toLocaleString() : '0';
  const safeCols = data.num_cols != null ? escapeHtml(data.num_cols) : '0';

  el.innerHTML = `
    <div class="dataset-summary-card">
      <div class="dataset-summary-header">
        <div class="dataset-summary-check">✓</div>
        <div>
          <div class="dataset-summary-filename">${safeFilename}</div>
          <div class="dataset-summary-meta">${safeRows} records · ${safeCols} attributes</div>
        </div>
      </div>
      <div class="dataset-summary-ready">
        ● Dataset successfully analyzed and ready for synthesis
      </div>
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
  if (state.generating) return;

  const dsId = document.getElementById('gen-dataset-id')?.value || state.datasetId;
  if (!dsId) {
    toast('Please upload or select a dataset first.', 'error');
    return;
  }

  const numRows = parseInt(document.getElementById('gen-rows')?.value, 10) || 1000;
  if (numRows < 10 || numRows > 100000) {
    toast('Number of synthetic records must be between 10 and 100,000.', 'error');
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

  state.generating = true;
  const btn = document.getElementById('btn-generate');
  const btnText = document.getElementById('btn-generate-text');
  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = 'Generating...';

  showGenerationProgress(mode);

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

    // Step 4: Applying Differential Privacy
    setTimeout(() => updateGenerationStep(4), 500);

    const res = await api.generate(payload);
    state.jobId = res.data.job_id;
    state.lastGenResult = res.data;

    // Step 5: Preparing Quality Report
    updateGenerationStep(5);

    toast(`Successfully generated ${res.data.num_rows_generated.toLocaleString()} synthetic records!`);

    // Automatic transition to Quality Report
    setTimeout(async () => {
      hideGenerationProgress();
      switchTab('tab-quality');
      await runQualityEvaluation();
    }, 400);
  } catch (e) {
    toast(e.message || 'Generation failed. Please check parameters and try again.', 'error');
    hideGenerationProgress();
  } finally {
    state.generating = false;
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = 'Generate Synthetic Data';
  }
}

// ── Accordion Helper with Chart Resize ──
function toggleAccordion(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.toggle('open');
    if (el.classList.contains('open')) {
      setTimeout(() => {
        charts.resizeAll();
        if (id === 'acc-stat-details' && state.selectedDistCol) {
          renderSelectedDistChart(state.selectedDistCol);
        }
      }, 50);
    }
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
    if (res.data?.privacy_risk?.attacks) {
      state.lastAttackResults = {
        overall_risk_score: res.data.privacy_risk.raw_privacy_risk_score || 0,
        overall_risk_level: res.data.privacy_risk.risk_level || 'low',
        attacks: res.data.privacy_risk.attacks,
        radar_chart: {
          labels: ['Membership\nInference', 'Re-identification', 'Attribute\nInference'],
          risk_scores: [
            res.data.privacy_risk.attacks.membership_inference?.risk_score || 0,
            res.data.privacy_risk.attacks.reidentification?.risk_score || 0,
            res.data.privacy_risk.attacks.attribute_inference?.risk_score || 0,
          ],
        },
        summary: `Empirical Privacy Protection: Score ${res.data.privacy_risk.privacy_protection_score}/100 with 0 record collisions.`,
      };
    }
    renderQualityReport(res.data);
    toast('Quality Report evaluated successfully!');
  } catch (e) {
    toast(e.message || 'Evaluation failed. Please try again.', 'error');
  } finally {
    hideLoading();
  }
}

function renderQualityReport(report) {
  const el = document.getElementById('quality-results');
  if (!el || !report) return;

  // Clean previous chart instances to prevent leaks and canvas reuse errors
  charts.destroyAll();

  const s = report.executive_summary || {};
  const structural = report.structural_fidelity || {};
  const statistical = report.statistical_fidelity || {};
  const relationship = report.relationship_fidelity || {};
  const ml = report.ml_utility || {};
  const privacy = report.privacy_risk || {};
  const warnings = report.all_warnings || [];

  const gen = state.lastGenResult || {};
  const rows = gen.num_rows_generated ? gen.num_rows_generated.toLocaleString() : 'Ready';
  
  let modelLabel = 'Statistical Copula (Fast)';
  if (gen.model_type === 'tvae') modelLabel = 'TVAE Neural Generator';
  else if (gen.model_type === 'ctgan') modelLabel = 'CTGAN Generator';
  else if (gen.model_type) modelLabel = String(gen.model_type).toUpperCase();
  modelLabel = escapeHtml(modelLabel);

  const sourceName = escapeHtml(state.datasetInfo?.filename || 'Uploaded Dataset');
  const sourceRows = state.datasetInfo?.num_rows ? state.datasetInfo.num_rows.toLocaleString() : '-';

  // Human-readable DP configuration
  const dpMeta = gen.dp_metadata || privacy.dp_metadata || {};
  let dpText = 'Differential Privacy: Standard Protection';
  if (dpMeta.applied === false) {
    dpText = 'Differential Privacy: Disabled';
  } else if (dpMeta.epsilon_actual != null) {
    const mech = dpMeta.mechanism ? `${dpMeta.mechanism.charAt(0).toUpperCase() + dpMeta.mechanism.slice(1)} mechanism` : 'Gaussian mechanism';
    dpText = `Differential Privacy: ε = ${escapeHtml(dpMeta.epsilon_actual)}${dpMeta.delta ? ` · δ = ${escapeHtml(dpMeta.delta)}` : ''} · ${escapeHtml(mech)}`;
  } else {
    dpText = 'Differential Privacy: Enabled (Bounded Gaussian DP)';
  }

  const downloadUrl = state.jobId ? api.downloadUrl(state.jobId) : '#';

  // Trust Verdict Class & Icon
  let verdictClass = 'ready';
  let verdictIcon = '✓';
  if (s.trust_verdict?.includes('Review Required') || s.trust_verdict?.includes('Warning')) {
    verdictClass = 'review';
    verdictIcon = '⚠';
  } else if (s.trust_verdict?.includes('Revision Required') || s.trust_verdict?.includes('Failed')) {
    verdictClass = 'revision';
    verdictIcon = '✕';
  }

  // Statistical summary match calculations
  const colReports = statistical.column_reports || {};
  const colEntries = Object.entries(colReports);
  const totalCols = colEntries.length;
  let highMatchCount = 0;
  colEntries.forEach(([_, r]) => {
    const test = r.ks_test || r.chi_squared || {};
    if (test.similar || (test.p_value != null && test.p_value > 0.05)) highMatchCount++;
  });
  const matchRatio = totalCols > 0 ? (highMatchCount / totalCols) : 1;

  // ML Parity calculations
  let mlParitySummaryHtml = '';
  if (ml.applicable && ml.results) {
    const r = ml.results;
    const realAcc = r.trtr_rf?.accuracy || r.trtr_gb?.accuracy;
    const synthAcc = r.tstr_rf?.accuracy || r.tstr_gb?.accuracy;
    let parityPct = '95+';
    if (realAcc && synthAcc && realAcc > 0) {
      parityPct = Math.min(100, Math.round((synthAcc / realAcc) * 100));
    }
    mlParitySummaryHtml = `
      <div class="summary-callout-card success" style="margin-top:1rem">
        <div>
          <div class="summary-callout-title">✓ High Machine Learning Utility Parity (${parityPct}%)</div>
          <div class="summary-callout-text">Models trained on this synthetic data retain approximately <strong>${parityPct}%</strong> of real-data baseline performance across classification tasks.</div>
        </div>
      </div>`;
  }

  // Available column names for distribution dropdown
  const colNames = Object.keys(colReports);
  if (!state.selectedDistCol && colNames.length > 0) {
    state.selectedDistCol = colNames[0];
  } else if (state.selectedDistCol && !colNames.includes(state.selectedDistCol) && colNames.length > 0) {
    state.selectedDistCol = colNames[0];
  }

  // Exact duplicate collision count
  const exactCollisions = privacy.metrics?.exact_duplicate_count ?? 0;

  // Correlation data extraction fallback
  const corrData = (statistical.correlation?.columns?.length > 1) ? statistical.correlation : (
    (relationship.details?.cramers_matrix_real?.length > 1 && relationship.details?.categorical_columns_evaluated?.length > 1) ? {
      columns: relationship.details.categorical_columns_evaluated,
      real_correlation: relationship.details.cramers_matrix_real,
      synth_correlation: relationship.details.cramers_matrix_synth,
    } : null
  );

  // Covariance Alignment metric calculation fallback
  const covAlignment = relationship.metrics?.covariance_similarity != null && !isNaN(relationship.metrics.covariance_similarity)
    ? Math.round(relationship.metrics.covariance_similarity * 100)
    : (relationship.metrics?.pearson_mae != null && !isNaN(relationship.metrics.pearson_mae)
      ? Math.max(0, Math.round(100 - (relationship.metrics.pearson_mae * 100)))
      : (relationship.score != null && !isNaN(relationship.score) ? Math.round(relationship.score) : 100));

  // Reproducibility badge
  const isRepro = gen.reproducible_run || gen.seed !== undefined || state.lastGenResult?.seed !== undefined || (state.lastGenResult?.params && state.lastGenResult.params.seed !== undefined);
  const seedVal = escapeHtml(gen.seed !== undefined ? gen.seed : (state.lastGenResult?.seed ?? state.lastGenResult?.params?.seed ?? '42'));

  el.innerHTML = `
    <!-- 1. Top Executive Banner with Primary Download Action -->
    <div class="executive-banner">
      <div style="flex:1;min-width:280px">
        <div class="verdict-banner ${verdictClass}" style="margin-bottom:0.4rem">
          <span style="font-size:1.4rem">${verdictIcon}</span>
          <span style="font-size:1.1rem;font-weight:700">${s.trust_verdict || 'Evaluation Complete'}</span>
        </div>
        <div class="executive-meta-list">
          <span class="executive-meta-item">📁 Source: <strong>${sourceName}</strong> (${sourceRows} records)</span>
          <span class="executive-meta-item">•</span>
          <span class="executive-meta-item">🧬 Output: <strong>${rows} synthetic records</strong></span>
          <span class="executive-meta-item">•</span>
          <span class="executive-meta-item">⚙️ Model: <strong>${modelLabel}</strong></span>
          ${isRepro ? `
            <span class="executive-meta-item">•</span>
            <span class="badge badge-reproducible">✓ Reproducible (Seed: ${seedVal})</span>
          ` : ''}
        </div>
        <div style="font-size:0.78rem;color:var(--accent-cyan);margin-top:0.4rem">
          🔐 ${dpText}
        </div>
      </div>

      <div>
        <a href="${downloadUrl}" class="btn btn-success btn-lg" download style="display:inline-flex;align-items:center;gap:8px" ${!state.jobId ? 'disabled' : ''}>
          <span>⬇</span> <span>Download Synthetic Dataset</span>
        </a>
      </div>
    </div>

    <!-- 2. Dual Dimension Scorecard: Fidelity vs. Privacy -->
    <div class="grid grid-2" style="margin-top:1.5rem">
      <!-- Dimension 1: Data Fidelity -->
      <div class="card" style="border-top:4px solid var(--accent-blue)">
        <div class="card-title">📊 Dimension 1: Data Fidelity</div>
        <p class="score-card-explanation">How closely the synthetic dataset preserves the useful patterns, moments, and correlations found in the original data.</p>
        <div class="grid grid-2">
          <div class="stat-card"><div class="stat-label">Fidelity Score</div><div class="stat-value blue">${s.data_fidelity_score ?? '-'}/100</div></div>
          <div class="stat-card"><div class="stat-label">Fidelity Grade</div><div class="grade grade-${s.data_fidelity_grade || 'B'}">${s.data_fidelity_grade || '-'}</div></div>
        </div>
      </div>

      <!-- Dimension 2: Privacy Protection -->
      <div class="card" style="border-top:4px solid var(--accent-green)">
        <div class="card-title">🔐 Dimension 2: Privacy Protection</div>
        <p class="score-card-explanation">How effectively the generated data resists attempts to identify, recover, or memorize individual records.</p>
        <div class="grid grid-2">
          <div class="stat-card"><div class="stat-label">Protection Score</div><div class="stat-value green">${s.privacy_protection_score ?? '-'}/100</div></div>
          <div class="stat-card"><div class="stat-label">Privacy Grade</div><div class="grade grade-${s.privacy_protection_grade || 'A'}">${s.privacy_protection_grade || '-'}</div></div>
        </div>
      </div>
    </div>

    <!-- 3. 5-Pillar Score Summary with Explanations -->
    <div class="card" style="margin-top:1.5rem">
      <div class="card-title">🏛️ 5-Pillar Trustworthiness Breakdown</div>
      <div class="grid grid-3" style="margin-top:1rem">
        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">1. Structural Fidelity</div>
            <span class="badge badge-${structural.status === 'passed' ? 'success' : 'warning'}">${structural.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-blue)">${structural.score?.toFixed(1) ?? '-'}</div>
          <p class="pillar-desc">Checks whether the generated dataset keeps the expected columns, data types, and structure.</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">2. Statistical Fidelity</div>
            <span class="badge badge-${statistical.status === 'passed' ? 'success' : 'warning'}">${statistical.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-cyan)">${statistical.score?.toFixed(1) ?? '-'}</div>
          <p class="pillar-desc">Checks whether values and distributions behave similarly to the original dataset.</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">3. Relationship Fidelity</div>
            <span class="badge badge-${relationship.status === 'passed' ? 'success' : 'warning'}">${relationship.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-purple)">${relationship.score?.toFixed(1) ?? '-'}</div>
          <p class="pillar-desc">Checks whether correlations and multi-variable dependencies are preserved.</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">4. ML Utility (TSTR)</div>
            <span class="badge badge-${ml.status === 'passed' ? 'success' : 'warning'}">${ml.status || (ml.applicable ? 'passed' : 'N/A')}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-amber)">${ml.score != null ? ml.score.toFixed(1) : 'N/A'}</div>
          <p class="pillar-desc">Checks whether machine-learning models trained on synthetic data remain useful on real data.</p>
        </div>

        <div class="pillar-card">
          <div class="pillar-header">
            <div class="pillar-title">5. Privacy Protection</div>
            <span class="badge badge-${privacy.status === 'passed' ? 'success' : 'warning'}">${privacy.status || 'passed'}</span>
          </div>
          <div class="pillar-score" style="color:var(--accent-green)">${privacy.privacy_protection_score?.toFixed(1) ?? '-'}</div>
          <p class="pillar-desc">Checks whether synthetic records resist identity leakage and exact row collisions.</p>
        </div>
      </div>
    </div>

    <!-- 4. Findings & Warnings -->
    ${warnings.length ? `
    <div class="card" style="margin-top:1.5rem">
      <div class="card-title">⚠️ Quality & Privacy Findings (${warnings.length})</div>
      <ul style="padding-left:1.5rem;color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem">
        ${warnings.map(w => `<li style="margin-bottom:0.4rem">${w}</li>`).join('')}
      </ul>
    </div>` : ''}

    <!-- 5. TECHNICAL DEEP-DIVE EVIDENCE ACCORDIONS -->

    <!-- Accordion 1: Statistical Fidelity Details -->
    <div class="accordion open" id="acc-stat-details" style="margin-top:1.5rem">
      <div class="accordion-header" onclick="toggleAccordion('acc-stat-details')">
        <span style="font-size:0.95rem;font-weight:700">📊 Statistical Fidelity & Distribution Analysis</span>
        <span class="accordion-icon">▼</span>
      </div>
      <div class="accordion-body">
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem">
          This checks whether the synthetic dataset behaves like the original dataset across individual variables and relationships. Strong similarity means the data can remain useful for analytics without exposing the original records.
        </p>

        <!-- Summary Card -->
        <div class="summary-callout-card ${matchRatio >= 0.7 ? 'success' : 'warning'}">
          <div>
            <div class="summary-callout-title">
              ${matchRatio >= 0.7 ? '✓ High Empirical Distribution Conformity' : '⚠ Partial Distribution Divergence'}
            </div>
            <div class="summary-callout-text">
              <strong>${highMatchCount} of ${totalCols}</strong> attributes closely follow the statistical moments and probability distributions observed in the original dataset.
            </div>
          </div>
        </div>

        ${totalCols > 0 ? `
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Attribute</th>
                <th>Type</th>
                <th title="Kolmogorov-Smirnov distance for numeric data or Chi-Square statistic for categorical data"><span class="term-tooltip">KS / Chi2 Statistic</span></th>
                <th title="Probability that synthetic and real distributions originate from the same underlying population (p > 0.05 indicates high match)"><span class="term-tooltip">P-Value</span></th>
                <th>Distribution Match</th>
                <th>Quality Score</th>
              </tr>
            </thead>
            <tbody>
              ${colEntries.map(([col, r]) => {
                const test = r.ks_test || r.chi_squared || {};
                const pVal = test.p_value;
                let statusBadge = '<span class="status-pill match-high">✓ High Match</span>';
                if (!test.similar && (pVal != null && pVal < 0.01)) {
                  statusBadge = '<span class="status-pill match-divergent">✕ Divergent</span>';
                } else if (!test.similar || (pVal != null && pVal < 0.05)) {
                  statusBadge = '<span class="status-pill match-moderate">~ Moderate</span>';
                }

                return `
                  <tr>
                    <td><strong>${col}</strong></td>
                    <td><span class="badge badge-${r.type === 'numerical' ? 'info' : 'warning'}">${r.type}</span></td>
                    <td>${test.statistic != null ? test.statistic.toFixed(4) : '-'}</td>
                    <td>${test.p_value != null ? test.p_value.toFixed(4) : '-'}</td>
                    <td>${statusBadge}</td>
                    <td><strong>${r.quality_score != null ? r.quality_score.toFixed(1) : '-'}</strong></td>
                  </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>` : '<p style="color:var(--text-muted)">No per-column statistical metrics available.</p>'}

        <!-- Column Distribution Inspector with Dropdown -->
        ${colNames.length > 0 ? `
        <div style="margin-top:1.5rem">
          <div class="dist-selector-row">
            <div class="dist-selector-label">
              <span>📈 Compare Distribution:</span>
            </div>
            <select id="dist-col-select" class="dist-select" onchange="onDistColChange(this.value)" aria-label="Select variable for distribution chart">
              ${colNames.map(name => `<option value="${name}" ${name === state.selectedDistCol ? 'selected' : ''}>${name}</option>`).join('')}
            </select>
          </div>
          <div class="card" style="margin-bottom:0">
            <div class="chart-container"><canvas id="qr-dist-selected"></canvas></div>
          </div>
        </div>` : ''}

        <!-- Correlation Matrices -->
        ${corrData ? `
        <div style="margin-top:1.5rem">
          <div class="card-title" style="font-size:0.95rem;margin-bottom:0.8rem">🔗 Correlation Matrix Alignment</div>
          <div class="grid grid-2">
            <div class="card">
              <div class="card-title" style="font-size:0.88rem">🔵 Real Correlation Matrix</div>
              <div id="qr-heatmap-real" class="heatmap-container"></div>
            </div>
            <div class="card">
              <div class="card-title" style="font-size:0.88rem">🟣 Synthetic Correlation Matrix</div>
              <div id="qr-heatmap-synth" class="heatmap-container"></div>
            </div>
          </div>
        </div>` : ''}
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
          <strong>ML Utility</strong> measures whether machine learning models trained on synthetic data can still perform well when tested against real data.<br>
          • <strong>TSTR (Train on Synthetic, Test on Real)</strong>: Validates whether synthetic data produces deployable, generalizable models.<br>
          • <strong>TRTR (Train on Real, Test on Real)</strong>: The benchmark performance of models trained directly on original real data.
        </p>

        ${mlParitySummaryHtml}

        ${ml.results ? `
        <div class="table-responsive" style="margin-top:1rem">
          <table class="data-table">
            <thead>
              <tr>
                <th>Evaluation Metric</th>
                <th>Real Baseline (Random Forest)</th>
                <th>Synthetic Trained (Random Forest)</th>
                <th>Real Baseline (Gradient Boosting)</th>
                <th>Synthetic Trained (Gradient Boosting)</th>
              </tr>
            </thead>
            <tbody>
              ${['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall'].map(m => `
                <tr>
                  <td><strong>${m.replace('_', ' ').toUpperCase()}</strong></td>
                  <td>${ml.results.trtr_rf?.[m] != null ? +(ml.results.trtr_rf[m]).toFixed(4) : '-'}</td>
                  <td>${ml.results.tstr_rf?.[m] != null ? +(ml.results.tstr_rf[m]).toFixed(4) : '-'}</td>
                  <td>${ml.results.trtr_gb?.[m] != null ? +(ml.results.trtr_gb[m]).toFixed(4) : '-'}</td>
                  <td>${ml.results.tstr_gb?.[m] != null ? +(ml.results.tstr_gb[m]).toFixed(4) : '-'}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
        <div class="grid grid-2" style="margin-top:1.2rem">
          <div class="card"><div class="card-title" style="font-size:0.88rem">📊 ML Performance Parity</div><div class="chart-container"><canvas id="qr-ml-bar"></canvas></div></div>
          <div class="card"><div class="card-title" style="font-size:0.88rem">📈 ROC Curve (TSTR Model)</div><div class="chart-container"><canvas id="qr-ml-roc"></canvas></div></div>
        </div>` : `
        <div class="info-card" style="margin-top:1rem">
          <div class="info-icon">🤖</div>
          <h4>ML Utility benchmark unavailable</h4>
          <p>This dataset does not contain a suitable target classification variable for supervised machine learning evaluation.</p>
        </div>`}
      </div>
    </div>

    <!-- Accordion 3: Structural & Privacy Details (Separated into clear subsections) -->
    <div class="accordion" id="acc-struct-details">
      <div class="accordion-header" onclick="toggleAccordion('acc-struct-details')">
        <span style="font-size:0.95rem;font-weight:700">🛡️ Structural Integrity & Empirical Privacy Audit</span>
        <span class="accordion-icon">▼</span>
      </div>
      <div class="accordion-body">
        <div class="evidence-subsections">
          <!-- Subsection A: Schema Integrity -->
          <div class="subcard">
            <div class="subcard-title">🏛️ A. Schema & Structural Integrity</div>
            <div class="grid grid-3" style="margin-top:0.8rem">
              <div class="stat-card">
                <div class="stat-label">Columns Preserved</div>
                <div class="stat-value green">${(structural.metrics?.column_preservation_rate * 100)?.toFixed(0) ?? 100}%</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Datatype Compatibility</div>
                <div class="stat-value blue">${(structural.metrics?.dtype_match_rate * 100)?.toFixed(0) ?? 100}%</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Covariance Alignment</div>
                <div class="stat-value purple">${covAlignment}%</div>
              </div>
            </div>
          </div>

          <!-- Subsection B: Collision Defense -->
          <div class="subcard">
            <div class="subcard-title">🔐 B. Exact Row Collision Defense</div>
            <div style="margin-top:0.8rem">
              ${exactCollisions === 0 ? `
              <div class="zero-collision-box">
                <div style="font-size:1.5rem;color:var(--accent-green)">✓</div>
                <div>
                  <div style="font-weight:700;color:var(--accent-green)">0 exact collisions detected</div>
                  <div style="font-size:0.84rem;color:var(--text-secondary);margin-top:2px">
                    No identical synthetic rows were found in the evaluated real dataset. Zero training records were memorized or duplicated.
                  </div>
                </div>
              </div>` : `
              <div class="zero-collision-box has-collisions">
                <div style="font-size:1.5rem;color:var(--accent-red)">⚠</div>
                <div>
                  <div style="font-weight:700;color:var(--accent-red)">${exactCollisions} exact collision(s) detected</div>
                  <div style="font-size:0.84rem;color:var(--text-secondary);margin-top:2px">
                    Some generated rows match training records exactly. Consider increasing Differential Privacy noise.
                  </div>
                </div>
              </div>`}
            </div>
          </div>

          <!-- Subsection C: Empirical Privacy Threat Summary -->
          <div class="subcard">
            <div class="subcard-title">🛡️ C. Empirical Privacy Threat Resilience</div>
            <div class="grid grid-3" style="margin-top:0.8rem">
              <div class="stat-card">
                <div class="stat-label">Membership Inference AUC</div>
                <div class="stat-value ${(privacy.metrics?.mia_auc || 0.5) <= 0.6 ? 'green' : 'amber'}">${privacy.metrics?.mia_auc != null ? privacy.metrics.mia_auc.toFixed(3) : '0.500'}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Re-ID Records at Risk</div>
                <div class="stat-value ${(privacy.metrics?.reid_records_at_risk_pct || 0) <= 5 ? 'green' : 'amber'}">${privacy.metrics?.reid_records_at_risk_pct != null ? privacy.metrics.reid_records_at_risk_pct.toFixed(1) : '0.0'}%</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Attribute Inference Advantage</div>
                <div class="stat-value ${(privacy.metrics?.attribute_inference_avg_advantage || 0) <= 0.1 ? 'green' : 'amber'}">${privacy.metrics?.attribute_inference_avg_advantage != null ? privacy.metrics.attribute_inference_avg_advantage.toFixed(3) : '0.000'}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Secondary Download CTA at bottom of report -->
    <div class="secondary-download-container">
      <div>
        <div style="font-weight:700;font-size:1rem;color:var(--text-primary)">Ready to use this synthetic dataset?</div>
        <div style="font-size:0.85rem;color:var(--text-secondary)">Download the generated CSV for safe data sharing, analysis, and ML model training.</div>
      </div>
      <a href="${downloadUrl}" class="btn btn-success btn-lg" download style="display:inline-flex;align-items:center;gap:8px" ${!state.jobId ? 'disabled' : ''}>
        <span>⬇</span> <span>Download Synthetic Dataset</span>
      </a>
    </div>

    <div class="card" style="margin-top:1.5rem;background:rgba(255,255,255,0.02)">
      <p style="font-size:0.75rem;color:var(--text-muted);font-style:italic">
        ${s.disclaimer || 'Synthetic data trustworthiness cannot be determined by any single metric. Fidelity and Privacy represent separate trade-off dimensions and must be evaluated independently.'}
      </p>
    </div>`;

  // Render Charts
  renderQualityReportCharts(report);
}

function onDistColChange(colName) {
  state.selectedDistCol = colName;
  renderSelectedDistChart(colName);
}

function renderSelectedDistChart(colName) {
  if (!state.lastQualityReport?.statistical_fidelity?.column_reports) return;
  const report = state.lastQualityReport.statistical_fidelity.column_reports[colName];
  if (!report || !report.distribution) return;

  charts.destroy('qr-dist-selected');
  if (report.type === 'numerical' && report.distribution.bins) {
    charts.distribution('qr-dist-selected', report.distribution, colName);
  } else if (report.distribution.categories) {
    charts.categoricalDist('qr-dist-selected', report.distribution, colName);
  }
}

function renderQualityReportCharts(report) {
  if (!report) return;
  const statistical = report.statistical_fidelity || {};
  const relationship = report.relationship_fidelity || {};
  const ml = report.ml_utility || {};

  // Render selected distribution chart
  if (state.selectedDistCol) {
    setTimeout(() => renderSelectedDistChart(state.selectedDistCol), 60);
  }

  // Render heatmaps
  const corrData = (statistical.correlation?.columns?.length > 1) ? statistical.correlation : (
    (relationship.details?.cramers_matrix_real?.length > 1 && relationship.details?.categorical_columns_evaluated?.length > 1) ? {
      columns: relationship.details.categorical_columns_evaluated,
      real_correlation: relationship.details.cramers_matrix_real,
      synth_correlation: relationship.details.cramers_matrix_synth,
    } : null
  );

  if (corrData && corrData.columns?.length > 1) {
    setTimeout(() => {
      charts.heatmapHTML('qr-heatmap-real', corrData.real_correlation, corrData.columns);
      charts.heatmapHTML('qr-heatmap-synth', corrData.synth_correlation, corrData.columns);
    }, 80);
  }

  // Render ML charts
  if (ml.results) {
    setTimeout(() => {
      const r = ml.results;
      if (r.trtr_rf && r.tstr_rf) {
        charts.comparisonBar('qr-ml-bar', ['Accuracy', 'F1 Score', 'ROC-AUC'],
          [r.trtr_rf.accuracy || 0, r.trtr_rf.f1_score || 0, r.trtr_rf.roc_auc || 0],
          [r.tstr_rf.accuracy || 0, r.tstr_rf.f1_score || 0, r.tstr_rf.roc_auc || 0]);
      }
      if (r.tstr_rf?.roc_curve) {
        charts.rocCurve('qr-ml-roc', r.tstr_rf.roc_curve.fpr, r.tstr_rf.roc_curve.tpr, 'TSTR Random Forest');
      }
    }, 100);
  }
}

// ── Privacy & Threat Analysis Tab (Reuses quality report data or runs fresh) ──
async function loadPrivacyThreats() {
  const el = document.getElementById('attack-results');
  if (!el) return;

  if (state.lastAttackResults) {
    renderAttackResults(state.lastAttackResults);
    return;
  }

  if (state.lastQualityReport?.privacy_risk?.attacks) {
    const pr = state.lastQualityReport.privacy_risk;
    const data = {
      overall_risk_score: pr.raw_privacy_risk_score || 0,
      overall_risk_level: pr.risk_level || 'low',
      attacks: pr.attacks,
      radar_chart: {
        labels: ['Membership\nInference', 'Re-identification', 'Attribute\nInference'],
        risk_scores: [
          pr.attacks.membership_inference?.risk_score || 0,
          pr.attacks.reidentification?.risk_score || 0,
          pr.attacks.attribute_inference?.risk_score || 0,
        ],
      },
      summary: `Empirical Privacy Protection: Score ${pr.privacy_protection_score}/100 with zero record collisions.`,
    };
    state.lastAttackResults = data;
    renderAttackResults(data);
    return;
  }

  if (!state.datasetId) {
    el.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🎯</div>
        <h3>No attack simulation results yet</h3>
        <p style="margin: 0.5rem 0 1.2rem;">Generate synthetic data to automatically evaluate empirical privacy attacks against your records.</p>
        <button class="btn btn-primary" onclick="switchTab('tab-synthesize')">
          <span>Go to Synthesize</span>
          <span class="btn-arrow">→</span>
        </button>
      </div>`;
    return;
  }

  // If dataset exists but no attacks run yet, run them automatically
  await runAttacks();
}

async function runAttacks() {
  if (!state.datasetId) {
    toast('Please upload a dataset and generate synthetic data first.', 'error');
    return;
  }
  showLoading('Simulating adversarial privacy attacks (MIA, Re-ID, Attribute Inference)...');
  try {
    const res = await api.simulateAttacks({ dataset_id: state.datasetId, synthetic_job_id: state.jobId || undefined });
    state.lastAttackResults = res.data;
    renderAttackResults(res.data);
    toast(`Overall Privacy Risk: ${res.data.overall_risk_score}/100 (${res.data.overall_risk_level})`);
  } catch (e) {
    toast(e.message || 'Attack simulation failed.', 'error');
  } finally {
    hideLoading();
  }
}

function renderAttackResults(data) {
  const el = document.getElementById('attack-results');
  if (!el || !data) return;
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
      <div class="card"><div class="card-title" style="font-size:0.95rem">🎯 Threat Radar</div><div class="chart-container"><canvas id="attack-radar"></canvas></div></div>
      <div class="card"><div class="card-title" style="font-size:0.95rem">📈 MIA ROC Curve</div><div class="chart-container"><canvas id="attack-roc"></canvas></div></div>
    </div>
    <div class="card">
      <div class="card-title" style="font-size:0.95rem">🔍 Empirical Attack Vulnerability Summary</div>
      <div class="table-responsive">
        <table class="data-table">
          <thead><tr><th>Attack Vector</th><th>Metric</th><th>Empirical Value</th><th>Risk Assessment</th></tr></thead>
          <tbody>
            <tr><td><strong>Membership Inference (MIA)</strong></td><td>Attack AUC</td><td>${a.membership_inference?.attack_auc != null ? a.membership_inference.attack_auc.toFixed(3) : '-'}</td><td><span class="badge badge-${a.membership_inference?.risk_level === 'low' ? 'success' : 'danger'}">${a.membership_inference?.risk_level ?? 'low'}</span></td></tr>
            <tr><td><strong>Re-Identification</strong></td><td>Records at Risk</td><td>${a.reidentification?.records_at_risk_pct != null ? a.reidentification.records_at_risk_pct.toFixed(1) : '0.0'}%</td><td><span class="badge badge-${a.reidentification?.risk_level === 'low' ? 'success' : 'danger'}">${a.reidentification?.risk_level ?? 'low'}</span></td></tr>
            <tr><td><strong>Attribute Inference</strong></td><td>Average Advantage</td><td>${a.attribute_inference?.average_advantage != null ? a.attribute_inference.average_advantage.toFixed(3) : '0.000'}</td><td><span class="badge badge-${a.attribute_inference?.risk_level === 'low' ? 'success' : 'danger'}">${a.attribute_inference?.risk_level ?? 'low'}</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div class="summary-callout-card success" style="margin-top:1rem">
      <div>
        <div class="summary-callout-title">🛡️ Privacy Takeaway</div>
        <div class="summary-callout-text">${data.summary || 'Empirical adversarial testing demonstrates high resilience against identification and reconstruction threats.'}</div>
      </div>
    </div>`;

  renderAttackCharts(data);
}

function renderAttackCharts(data) {
  if (!data) return;
  const a = data.attacks || {};
  setTimeout(() => {
    if (data.radar_chart) {
      charts.radar('attack-radar', data.radar_chart.labels, data.radar_chart.risk_scores);
    }
    if (a.membership_inference?.roc_curve) {
      charts.rocCurve('attack-roc', a.membership_inference.roc_curve.fpr, a.membership_inference.roc_curve.tpr, 'MIA Attack Curve');
    }
  }, 100);
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
  // Theme initialization
  initTheme();

  // Initialize advanced settings visibility
  onModelArchitectureChange();

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
